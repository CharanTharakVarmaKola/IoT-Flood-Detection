import csv
import json
import os
import time
import uuid

import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "flood/zone1/sensors"
CSV = "flood_data.csv"

FIELDS = [
    "ts",
    "level_cm",
    "rate_cm_min",
    "rain_adc",
    "rain_status",
    "temp_c",
    "hum",
    "water_adc",
    "sensor_ok",
    "uptime_s",
]


def main():
    new_file = not os.path.exists(CSV) or os.path.getsize(CSV) == 0
    with open(CSV, "a", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        rows = 0

        def on_connect(client, userdata, flags, rc, properties=None):
            if rc != 0:
                print(f"MQTT connection failed rc={rc}")
                return
            client.subscribe(TOPIC, qos=0)
            print(f"Collecting REAL ESP32 data from {TOPIC} -> {CSV}")
            print("Ctrl+C to stop.")

        def on_message(client, userdata, msg):
            nonlocal rows
            try:
                d = json.loads(msg.payload.decode("utf-8"))
                if not d.get("sensor_ok", False):
                    return

                writer.writerow({
                    "ts": f"{time.time():.4f}",
                    "level_cm": f"{float(d['level_cm']):.4f}",
                    "rate_cm_min": f"{float(d['rate_cm_min']):.4f}",
                    "rain_adc": f"{float(d['rain_adc']):.4f}",
                    "rain_status": str(d.get("rain_status", "Unknown")),
                    "temp_c": f"{float(d['temp_c']):.4f}",
                    "hum": f"{float(d['hum']):.4f}",
                    "water_adc": f"{float(d.get('water_adc', 0)):.4f}",
                    "sensor_ok": bool(d.get("sensor_ok", False)),
                    "uptime_s": f"{float(d.get('uptime_s', 0)):.4f}",
                })
                f.flush()
                rows += 1
                print(
                    f"\rrows={rows:4d} | level={float(d['level_cm']):.2f} cm | "
                    f"rate={float(d['rate_cm_min']):+.2f} cm/min | "
                    f"water_adc={int(float(d.get('water_adc', 0)))} | "
                    f"rain={d.get('rain_status', 'Unknown')} "
                    f"({int(float(d['rain_adc']))}) | "
                    f"T={float(d['temp_c']):.1f}C H={float(d['hum']):.1f}%",
                    end="",
                    flush=True,
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                return

        client = mqtt.Client(client_id=f"flood-collector-{uuid.uuid4().hex[:8]}")
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(BROKER, PORT, 60)
        client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCollection stopped.")
