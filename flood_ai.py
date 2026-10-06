"""
Flood Station — laptop-side AI + dashboard.

Commands:
    python flood_ai.py validate     validate the real training dataset
    python flood_ai.py train-risk   train the genuine future-risk classifier
    python flood_ai.py serve        run live Chronos-2 forecasting + dashboard

AI design:
  1) Chronos-2 (pretrained Hugging Face time-series foundation model)
     forecasts the water level 5 minutes ahead.
  2) A RandomForest classifier is trained on the REAL flood_data.csv to
     predict the observed future 0–4 cm risk class within the next 5 min.

There is no synthetic sensor stream and no rule-based value presented as AI.
Risk labels are created from future measured water levels only for training;
the live classifier receives only information available at the current time.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

# ----------------------------------------------------------------------
# Paths / configuration
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BROKER = "localhost"
PORT = 1883
MQTT_USER = ""
MQTT_PW = ""

T_SENSORS = "flood/zone1/sensors"
T_PRED = "flood/zone1/prediction"
T_CMD = "flood/zone1/command"

CSV = ROOT / "flood_data.csv"
RISK_MODEL = ROOT / "risk_model.joblib"
RISK_META = ROOT / "risk_model_meta.json"
DASHBOARD = ROOT / "dashboard.html"
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8000

TANK_CM = 4.0
WARN_CM = 2.0
DANGER_CM = 3.0
HORIZON_MIN = 5
SAMPLE_SEC = 5
HORIZON_STEPS = HORIZON_MIN * 60 // SAMPLE_SEC  # 60 steps

FEATURES = ["level_cm", "rate_cm_min", "rain_adc", "temp_c", "hum"]
RISK_FEATURES = FEATURES
RISK_CLASSES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
STALE_SEC = 20
SESSION_GAP_SEC = 60
MIN_CONTEXT_SAMPLES = 24        # 2 minutes of real station history before first forecast
MAX_CONTEXT_SAMPLES = 120       # 10 minutes of context
PREDICT_EVERY_SEC = 15          # do not hammer a CPU-only foundation model

state_lock = threading.Lock()

state = {
    "level_cm": 0.0,
    "rate_cm_min": 0.0,
    "rain_adc": 4095,
    "rain_status": "Dry",
    "predicted_cm": 0.0,
    "predict_src": "unavailable",
    "horizon_min": HORIZON_MIN,
    "risk": "UNKNOWN",
    "alert": "Unknown",
    "pump": "OFF",
    "pump_mode": "auto",
    "temp_c": 0.0,
    "hum": 0.0,
    "tank_cm": TANK_CM,
    "warn_cm": WARN_CM,
    "danger_cm": DANGER_CM,
    "sensor_ok": False,
    "mqtt": False,
    "uptime_s": 0.0,
    "station_online": False,
}

history = collections.deque(maxlen=MAX_CONTEXT_SAMPLES)
last_message_wall = 0.0
last_uptime = None
latest_prediction = None


# ----------------------------------------------------------------------
# MQTT helper — avoids the paho callback-version deprecation warning
# ----------------------------------------------------------------------
def make_client(client_id: str):
    try:
        return mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
    except (AttributeError, TypeError):
        return mqtt.Client(client_id=client_id)


def connect(client_id: str):
    client = make_client(client_id)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PW)
    client.connect(BROKER, PORT, 60)
    return client


def rain_status(adc: float) -> str:
    if adc >= 3000:
        return "Dry"
    if adc < 1500:
        return "Heavy"
    return "Light"


def clean_sensor(d: dict) -> dict | None:
    try:
        out = {k: float(d[k]) for k in FEATURES}
    except (KeyError, TypeError, ValueError):
        return None
    out["sensor_ok"] = bool(d.get("sensor_ok", False))
    out["water_adc"] = float(d.get("water_adc", 0))
    out["uptime_s"] = float(d.get("uptime_s", 0))
    out["rain_status"] = str(d.get("rain_status") or rain_status(out["rain_adc"]))
    out["pump"] = str(d.get("pump", state.get("pump", "OFF")))
    out["pump_mode"] = str(d.get("pump_mode", state.get("pump_mode", "auto")))
    return out


def update_live_state(d: dict) -> bool:
    global last_message_wall, last_uptime
    clean = clean_sensor(d)
    if clean is None or not clean["sensor_ok"]:
        return False

    # A reboot or a long MQTT gap starts a new live sequence. This prevents
    # Chronos from treating two unrelated sessions as continuous history.
    reset_history = False
    with state_lock:
        if last_uptime is not None and clean["uptime_s"] + 1 < last_uptime:
            reset_history = True
        if last_message_wall and time.time() - last_message_wall > SESSION_GAP_SEC:
            reset_history = True

        if reset_history:
            history.clear()

        # Regular 5-second model steps are based on incoming sample order.
        # Small network jitter therefore does not break Chronos' frequency
        # validation requirements.
        history.append({k: clean[k] for k in FEATURES})
        last_uptime = clean["uptime_s"]
        last_message_wall = time.time()

        state.update({
            "level_cm": clean["level_cm"],
            "rate_cm_min": clean["rate_cm_min"],
            "rain_adc": clean["rain_adc"],
            "rain_status": clean["rain_status"],
            "temp_c": clean["temp_c"],
            "hum": clean["hum"],
            "pump": clean["pump"],
            "pump_mode": clean["pump_mode"],
            "sensor_ok": True,
            "mqtt": True,
            "uptime_s": clean["uptime_s"],
            "station_online": True,
        })
    return True


# ----------------------------------------------------------------------
# Risk model — trained from future measured outcomes
# ----------------------------------------------------------------------
def future_risk_class(future_max: float) -> str:
    if future_max < WARN_CM:
        return "LOW"
    if future_max < DANGER_CM:
        return "MODERATE"
    if future_max < 3.5:
        return "HIGH"
    return "CRITICAL"


def build_risk_training_data(csv_path: Path):
    import pandas as pd

    if not csv_path.exists():
        raise RuntimeError(f"{csv_path.name} not found")

    df = pd.read_csv(csv_path)
    missing = [c for c in ["ts", *RISK_FEATURES, "sensor_ok"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {csv_path.name}: {missing}")

    df = df.copy()
    for c in ["ts", *RISK_FEATURES]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sensor_ok"] = df["sensor_ok"].astype(str).str.lower().eq("true")
    df = df.dropna(subset=["ts", *RISK_FEATURES])
    df = df[df["sensor_ok"]]
    df = df.sort_values("ts").reset_index(drop=True)

    # Session boundary: a gap > 60 s is never crossed by a training example.
    df["session_id"] = (df["ts"].diff().fillna(0) > SESSION_GAP_SEC).cumsum()

    X_rows, y_rows, groups = [], [], []
    for sid, g in df.groupby("session_id", sort=True):
        g = g.reset_index(drop=True)
        t = g["ts"].to_numpy(dtype=float)
        levels = g["level_cm"].to_numpy(dtype=float)

        for i in range(len(g)):
            future_mask = (t > t[i]) & (t <= t[i] + HORIZON_MIN * 60)
            idx = np.flatnonzero(future_mask)
            if len(idx) == 0:
                continue
            # Require the session to extend close to the complete 5-min horizon.
            if t[idx[-1]] - t[i] < 240:
                continue
            future_max = float(np.max(levels[idx]))
            X_rows.append(g.loc[i, RISK_FEATURES].to_numpy(dtype=float))
            y_rows.append(future_risk_class(future_max))
            groups.append(int(sid))

    if not X_rows:
        raise RuntimeError("No complete 5-minute future-labelled samples could be created.")

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=object)
    groups = np.asarray(groups, dtype=int)
    return X, y, groups


def cmd_train_risk(csv_path: Path = CSV):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, classification_report

    X, y, groups = build_risk_training_data(csv_path)
    classes_present = [c for c in RISK_CLASSES if c in set(y)]
    print(f"Risk-training samples: {len(y)}")
    print("Observed future-risk classes:", {c: int(np.sum(y == c)) for c in RISK_CLASSES if c in set(y)})

    if len(classes_present) < 2:
        raise RuntimeError(
            "The real dataset does not contain at least two future-risk classes. "
            "Collect another rise/fall session before training the classifier."
        )

    unique_groups = sorted(set(groups.tolist()))
    test_mask = None
    if len(unique_groups) >= 2:
        last_group = unique_groups[-1]
        test_mask = groups == last_group

    metrics = {"balanced_accuracy": None, "classification_report": None, "test_rows": 0}
    if test_mask is not None and test_mask.any() and (~test_mask).sum() >= 20:
        train_mask = ~test_mask
        model_test = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model_test.fit(X[train_mask], y[train_mask])
        pred = model_test.predict(X[test_mask])
        metrics["test_rows"] = int(test_mask.sum())
        train_classes = set(model_test.classes_.tolist())
        unseen_test = sorted(set(y[test_mask]) - train_classes)
        if unseen_test:
            print("Hold-out contains classes not seen during training:", unseen_test)
            print("Hold-out metrics are not reported as a valid generalization score.")
        else:
            metrics["balanced_accuracy"] = float(
                balanced_accuracy_score(y[test_mask], pred)
            )
            metrics["classification_report"] = classification_report(
                y[test_mask], pred, zero_division=0, output_dict=True
            )
            print(f"Held-out balanced accuracy: {metrics['balanced_accuracy']:.3f}")

    # Final demonstration model uses all labelled real data.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    bundle = {
        "model": model,
        "features": RISK_FEATURES,
        "classes": model.classes_.tolist(),
        "label_definition": {
            "LOW": "future max level < 2.0 cm within 5 min",
            "MODERATE": "future max level >= 2.0 and < 3.0 cm within 5 min",
            "HIGH": "future max level >= 3.0 and < 3.5 cm within 5 min",
            "CRITICAL": "future max level >= 3.5 cm within 5 min",
        },
        "training_rows": len(y),
        "sessions": unique_groups,
        "metrics": metrics,
    }
    joblib.dump(bundle, RISK_MODEL)
    RISK_META.write_text(json.dumps({
        k: v for k, v in bundle.items() if k != "model"
    }, indent=2))
    print(f"Saved {RISK_MODEL.name}")


def load_risk_model():
    if not RISK_MODEL.exists():
        return None
    try:
        import joblib
        bundle = joblib.load(RISK_MODEL)
        if not isinstance(bundle, dict) or "model" not in bundle:
            return None
        return bundle
    except Exception as exc:
        print(f"Risk model load failed: {exc}")
        return None


def predict_risk(bundle, clean: dict) -> str:
    if not bundle:
        return "UNKNOWN"
    try:
        x = [[float(clean[k]) for k in RISK_FEATURES]]
        return str(bundle["model"].predict(x)[0])
    except Exception as exc:
        print(f"Risk prediction error: {exc}")
        return "UNKNOWN"


def alert_of(risk: str) -> str:
    return {
        "LOW": "Normal",
        "MODERATE": "Warning",
        "HIGH": "Warning",
        "CRITICAL": "Danger",
    }.get(risk, "Unknown")


# ----------------------------------------------------------------------
# Chronos-2 inference
# ----------------------------------------------------------------------
class ChronosEngine:
    def __init__(self, risk_bundle, mqtt_client):
        self.risk_bundle = risk_bundle
        self.mqtt_client = mqtt_client
        self.pipeline = None
        self.last_prediction_monotonic = 0.0
        self.busy = False
        self.stop_event = threading.Event()

        try:
            from chronos import Chronos2Pipeline
        except Exception as exc:
            raise RuntimeError(
                "Chronos-2 is not importable in this Python environment. "
                "Install chronos-forecasting in the same venv."
            ) from exc

        print("Loading pretrained amazon/chronos-2 on CPU...")
        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="cpu",
        )
        print("Chronos-2 ready — zero-shot forecasting enabled.")

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _snapshot_history(self):
        with state_lock:
            return list(history)

    def _worker(self):
        while not self.stop_event.is_set():
            now = time.monotonic()
            if not self.busy and now - self.last_prediction_monotonic >= PREDICT_EVERY_SEC:
                rows = self._snapshot_history()
                if len(rows) >= MIN_CONTEXT_SAMPLES:
                    self.busy = True
                    try:
                        self._predict(rows)
                    except Exception as exc:
                        print(f"\nChronos-2 forecast error: {exc}")
                    finally:
                        self.last_prediction_monotonic = time.monotonic()
                        self.busy = False
            time.sleep(1.0)

    def _predict(self, rows):
        import pandas as pd

        rows = rows[-MAX_CONTEXT_SAMPLES:]
        now_ts = pd.Timestamp.now(tz="UTC").floor("s")
        timestamps = pd.date_range(
            end=now_ts,
            periods=len(rows),
            freq=f"{SAMPLE_SEC}s",
        )
        df = pd.DataFrame(rows)
        df.insert(0, "timestamp", timestamps)
        df.insert(0, "id", "zone1")

        pred_df = self.pipeline.predict_df(
            df,
            prediction_length=HORIZON_STEPS,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="level_cm",
            freq=f"{SAMPLE_SEC}s",
        )
        pred = float(pred_df["predictions"].iloc[-1])
        pred = max(0.0, min(TANK_CM, pred))

        with state_lock:
            current_clean = {
                "level_cm": state["level_cm"],
                "rate_cm_min": state["rate_cm_min"],
                "rain_adc": state["rain_adc"],
                "temp_c": state["temp_c"],
                "hum": state["hum"],
                "sensor_ok": state["sensor_ok"],
                "rain_status": state["rain_status"],
                "pump": state["pump"],
                "pump_mode": state["pump_mode"],
                "uptime_s": state["uptime_s"],
            }

        risk = predict_risk(self.risk_bundle, current_clean)
        alert = alert_of(risk)
        prediction = {
            "predicted_level_cm": round(pred, 2),
            "horizon_min": HORIZON_MIN,
            "risk": risk,
            "alert": alert,
            "model": "chronos-2",
            "ts": time.time(),
        }

        with state_lock:
            state.update({
                "predicted_cm": prediction["predicted_level_cm"],
                "predict_src": "chronos2",
                "horizon_min": HORIZON_MIN,
                "risk": risk,
                "alert": alert,
            })

        self.mqtt_client.publish(T_PRED, json.dumps(prediction), qos=0, retain=False)
        print(
            f"\rChronos-2: current={current_clean['level_cm']:.2f} cm -> "
            f"+{HORIZON_MIN} min={pred:.2f} cm | risk={risk:<8}",
            end="",
            flush=True,
        )


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
def dashboard_payload():
    with state_lock:
        out = dict(state)
        out["history"] = list(history)
        last = last_message_wall

    out["station_online"] = bool(last and (time.time() - last <= STALE_SEC))
    if not out["station_online"]:
        out["mqtt"] = False
    return out


class DashboardHandler(BaseHTTPRequestHandler):
    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/data":
            self._send(200, "application/json; charset=utf-8", json.dumps(dashboard_payload()).encode())
            return

        if path in ("/", "/dashboard.html"):
            if not DASHBOARD.exists():
                self._send(404, "text/plain; charset=utf-8", b"dashboard.html not found")
                return
            self._send(200, "text/html; charset=utf-8", DASHBOARD.read_bytes())
            return

        self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        if u.path != "/api/pump":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        mode = (parse_qs(u.query).get("mode") or [""])[0]
        if mode not in ("auto", "on", "off"):
            self._send(400, "application/json", b'{"error":"invalid pump mode"}')
            return
        with state_lock:
            state["pump_mode"] = mode
        try:
            PUBLISH_CLIENT.publish(T_CMD, json.dumps({"pump_mode": mode}), qos=0, retain=False)
        except Exception:
            pass
        self._send(200, "application/json", json.dumps({"pump_mode": mode}).encode())

    def log_message(self, _format, *args):
        return


PUBLISH_CLIENT = None


def start_server():
    server = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"Dashboard: http://localhost:{HTTP_PORT}/")
    return server


# ----------------------------------------------------------------------
# validate
# ----------------------------------------------------------------------
def cmd_validate(csv_path: Path = CSV):
    import pandas as pd

    if not csv_path.exists():
        raise SystemExit(f"{csv_path.name} not found")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    if "ts" in df:
        gaps = pd.to_numeric(df["ts"], errors="coerce").diff()
        print(f"Median sample gap: {gaps.dropna().median():.2f} s")
        print(f"Long gaps (>60 s): {(gaps > 60).sum()}")
    print(f"Level range: {df['level_cm'].min():.2f}–{df['level_cm'].max():.2f} cm")
    print(f"Rain ADC range: {df['rain_adc'].min():.0f}–{df['rain_adc'].max():.0f}")
    print("Rain status counts:")
    print(df["rain_status"].value_counts().to_string())


# ----------------------------------------------------------------------
# Serve
# ----------------------------------------------------------------------
def cmd_serve():
    global PUBLISH_CLIENT

    risk_bundle = load_risk_model()
    if risk_bundle:
        print(
            f"Loaded risk model: {len(risk_bundle['classes'])} classes, "
            f"{risk_bundle['training_rows']} training rows."
        )
    else:
        print("Risk model not found. Live risk will remain UNKNOWN until train-risk succeeds.")

    client = connect("flood-ai-backend")
    PUBLISH_CLIENT = client

    def on_msg(c, _userdata, msg):
        try:
            d = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return
        update_live_state(d)

    client.on_message = on_msg
    client.subscribe(T_SENSORS, qos=0)

    engine = ChronosEngine(risk_bundle, client)
    start_server()

    print(f"Forecasting +{HORIZON_MIN} min with pretrained Chronos-2 (real sensor history only).")
    print(f"MQTT: {T_SENSORS} -> {T_PRED}")
    print("Press Ctrl+C to stop.")

    try:
        client.loop_forever()
    finally:
        engine.stop_event.set()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("train-risk")
    sub.add_parser("serve")
    args = parser.parse_args()

    if args.cmd == "validate":
        cmd_validate()
    elif args.cmd == "train-risk":
        cmd_train_risk()
    elif args.cmd == "serve":
        cmd_serve()


if __name__ == "__main__":
    main()
