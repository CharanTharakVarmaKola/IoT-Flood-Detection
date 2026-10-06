/* =====================================================================
   IoT Early Flood Detection & Avoidance — ESP32 edge node
   ---------------------------------------------------------------------
   The ESP32 owns the hardware only. It measures, filters, and acts on
   AI decisions received from the laptop — it does not invent an AI
   decision of its own. No web server, no HTML, no dashboard code, no
   HC-SR04 code, and no LED code live in this file.

     ESP32  --[ flood/zone1/sensors    ]-->  laptop (flood_ai.py)
     ESP32  <--[ flood/zone1/prediction ]--  laptop (LSTM forecast + AI risk)
     ESP32  <--[ flood/zone1/command    ]--  laptop (pump mode button)

   Hardware as built:
     PCB water-level probe   VCC=3.3V  GND=GND  AO=GPIO34
     Rain sensor              AO=GPIO35
     DHT11                        =GPIO4
     Buzzer                       =GPIO25
     Relay IN                     =GPIO23  ->  5V pump, external supply

   The HC-SR04 is NOT used. It is not required for this system to run.

   PRIMARY SENSOR CHANGE — read this before wiring anything:
   The probe is only ~4 cm long, so "water level" here is a calibrated
   0.0–4.0 cm DEMONSTRATION indicator, not a real 0–30 cm depth gauge.
   0 cm = dry, 4 cm = probe fully submerged. Rising water is shown by
   physically raising the level on the probe, not by any code trick.

   AI SEPARATION — the most important rule in this file:
   This node measures level, rate of rise, rain, temperature, humidity.
   It does NOT compute a flood risk itself and does NOT fabricate a
   straight-line "prediction" and call it AI. The future water level
   comes only from the LSTM running in flood_ai.py, delivered over
   flood/zone1/prediction. If nothing fresh has arrived, this node
   reports predict_src = "unavailable" and risk = "UNKNOWN" — it never
   guesses on the AI's behalf.

   Arduino IDE: Tools > Board > "ESP32 Dev Module".
   PlatformIO:  framework = arduino, board = esp32dev (same file, no
                PlatformIO-specific APIs are used anywhere below).
   Libraries:   PubSubClient (Nick O'Leary)
                DHT sensor library (Adafruit) + Adafruit Unified Sensor
   ===================================================================== */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>

/* ===================== 1. WI-FI / MQTT CONFIGURATION ================ */

const char* WIFI_SSID = "Balaji hostel second floor";
const char* WIFI_PASS = "8838329866";

const char* MQTT_HOST = "192.168.0.4";   // laptop's IP on the same Wi-Fi
const int   MQTT_PORT = 1883;
const char* MQTT_USER = "";              // leave "" for an open broker
const char* MQTT_PASS = "";

/* ===================== 2. CALIBRATION CONSTANTS ====================== */

// ---- PCB water-level probe: piecewise-linear calibration ------------
// These anchor points were measured directly from the physical probe
// (filtered ADC value observed at each known physical level). This is
// a PROTOTYPE calibration for a ~4 cm probe, not a datasheet constant —
// re-measure and edit these five numbers if you change the probe.
//
// Empirical calibration from the completed rising/falling tests.
// This is a prototype 0–4 cm indicator for this exact probe/water setup.
// The upper range is compressed, so the dense anchors are intentional.
//   physical level (cm):  0.0   0.5   1.0   2.0   2.5   3.0   3.5   4.0
//   settled ADC approx:      0   520  1050  1665  1755  1870  1960  2060
const int   CAL_ADC[]   = {    0,   520,  1050,  1665,  1755,  1870,  1960,  2060 };
const float CAL_LEVEL[] = {  0.0,  0.5,   1.0,   2.0,   2.5,   3.0,   3.5,   4.0 };
const int   CAL_POINTS  = 8;

const float TANK_CM = 4.0;    // full demonstration range of this probe

// Demonstration thresholds inside the 0–4 cm prototype range.
// These are prototype cues for this demo rig, NOT real-world flood
// levels — relabel them if you present this to someone unfamiliar with
// the probe's scale.
const float WARN_LEVEL_CM     = 2.0;
const float DANGER_LEVEL_CM   = 3.0;
const float PUMP_ON_LEVEL_CM  = 2.5;   // local safety gate for the pump
const float PUMP_OFF_LEVEL_CM = 1.0;   // hysteresis gap below ON

// ---- Water-sensor validity heuristics --------------------------------
// A resistive probe has no "timeout" like an ultrasonic sensor, so
// "sensor problem" has to be inferred instead of measured directly.
// Two heuristics, both DEMONSTRATION-level, not datasheet specs:
//   1. a reading far above the highest calibrated anchor usually means
//      the probe is floating/disconnected, not "very wet"
//   2. raw samples that jump around wildly inside one burst usually
//      mean a bad/noisy connection, not a real fast splash
const int ADC_FAULT_HIGH = 4095;   // pegged ADC: treat as sensor fault
const int ADC_SPREAD_MAX = 200;    // empirical upper bound from calibration logs
const int DRY_ADC_THRESHOLD = 100; // hard dry reset; avoids slow wet->dry decay
const float EMA_ALPHA     = 0.75;  // responsive smoothing after the 11-sample median
const float LEVEL_HYST_CM = 0.08;  // suppress tiny display/rate chatter

// ---- Rain sensor calibration ------------------------------------------
// Dry board reads high, soaked board reads low. Record your own values.
const int RAIN_DRY_ABOVE   = 3000;   // >= this  -> Dry
const int RAIN_HEAVY_BELOW = 1500;   // <  this  -> Heavy ; between -> Light

// ---- Behaviour knobs ---------------------------------------------------
const int  FORECAST_HORIZON_MIN = 5;      // must match HORIZON_MIN in flood_ai.py
const bool RELAY_ACTIVE_LOW     = true;   // most blue relay modules are
const unsigned long PUMP_MAX_RUN_MS  = 120000;  // protect a dry-running pump
const unsigned long PUMP_MIN_REST_MS = 30000;

/* ===================== 3. PIN DEFINITIONS ============================ */

#define PIN_WATER  34   // PCB water-level probe, analog output
#define PIN_RAIN   35   // rain sensor, analog output
#define PIN_DHT     4
#define PIN_BUZZER 25
#define PIN_RELAY  23

#define DHTTYPE DHT11

const unsigned long SAMPLE_MS     = 5000;   // sensor sample/publish period
const unsigned long PUBLISH_MS    = 5000;   // flood_ai.py expects ~5 s
const unsigned long PRED_VALID_MS = 30000;  // AI forecast goes stale after this
const int  WIN = 12;                        // 60 s of AI history
const int  RATE_WIN = 6;                    // 25 s slope window; avoids 60 s lag
const float RATE_ZERO_BAND = 0.12;           // cm/min below this -> steady
const float RATE_MAX_CM_MIN = 8.0;           // prototype plausibility cap

// Topic names — must match flood_ai.py exactly.
const char* T_SENSORS = "flood/zone1/sensors";
const char* T_PRED    = "flood/zone1/prediction";   // subscribed
const char* T_CMD     = "flood/zone1/command";      // subscribed

/* ===================== 4. GLOBAL STATE ================================ */

DHT dht(PIN_DHT, DHTTYPE);
WiFiClient   net;
PubSubClient mqtt(net);

float         levelBuf[WIN];
unsigned long timeBuf[WIN];
int   bufCount = 0, bufHead = 0;

float g_level = 0, g_rate = 0, g_predicted = 0;
float g_temp = NAN, g_hum = NAN;
int   g_rainAdc = 4095, g_waterAdc = 0;
bool  g_sensorOk = false;
float g_waterAdcEma = -1;
int g_dryBaselineAdc = 0;                 // -1 = "not seeded yet"
String g_rain = "Dry";

// AI-only risk state. Defaults reflect "nothing received from the AI
// yet" — never a guessed severity.
String g_predSrc = "unavailable";
String g_risk    = "UNKNOWN";
String g_alert   = "Unknown";

// forecast / risk arriving from the laptop's genuine LSTM
float  lstmLevel = 0;
String lstmRisk  = "";
unsigned long lstmAt = 0;

// pump
String pumpMode = "auto";                 // auto | on | off
bool   pumpOn = false;
unsigned long pumpChangedAt = 0;

unsigned long tSample = 0, tPublish = 0, tBuzz = 0;
bool buzzState = false;

/* ===================== 5–7. WATER-LEVEL SENSOR: READ, FILTER, CALIBRATE */

// Raw burst of ADC samples -> median (out) and spread (out). The median
// rejects an isolated spike; the spread is later used as a fault signal
// (see ADC_SPREAD_MAX above) since a healthy connection reads fairly
// steady across a handful of samples taken milliseconds apart.
void sampleWaterAdcRaw(int &medianOut, int &spreadOut) {
  const int N = 11;
  int v[N];
  for (int i = 0; i < N; i++) { v[i] = analogRead(PIN_WATER); delay(5); }
  for (int i = 1; i < N; i++) {                 // insertion sort
    int k = v[i]; int j = i - 1;
    while (j >= 0 && v[j] > k) { v[j + 1] = v[j]; j--; }
    v[j + 1] = k;
  }
  medianOut = v[N / 2];
  spreadOut = v[N - 1] - v[0];
}
void measureDryBaseline() {
  const int N = 15;
  long sum = 0;

  Serial.println("Measuring dry water-sensor baseline...");
  delay(1000);

  for (int i = 0; i < N; i++) {
    int v = analogRead(PIN_WATER);
    sum += v;
    delay(100);
  }

  g_dryBaselineAdc = (int)((sum / (float)N) + 0.5f);

  Serial.print("Dry baseline ADC: ");
  Serial.println(g_dryBaselineAdc);
}
// Piecewise-linear interpolation across the CAL_ADC / CAL_LEVEL anchor
// table. Chosen over a single straight-line fit because the probe's
// measured response is NOT linear across its range (see the anchor
// values above) — interpolating segment-by-segment tracks the actual
// measured curve instead of averaging over it.
float adcToLevelCm(int adc) {
  // Runtime dry-zero: anything at or below the measured dry baseline
  // is treated as 0 cm.
  if (adc <= g_dryBaselineAdc) return 0.0f;

  for (int i = 1; i < CAL_POINTS; i++) {
    // The first calibration segment starts at the runtime dry baseline.
    int lowAdc = (i == 1) ? g_dryBaselineAdc : CAL_ADC[i - 1];
    float lowLevel = (i == 1) ? CAL_LEVEL[0] : CAL_LEVEL[i - 1];

    int highAdc = CAL_ADC[i];
    float highLevel = CAL_LEVEL[i];

    if (adc <= highAdc) {
      float span = (float)(highAdc - lowAdc);

      if (span <= 0.0f) return lowLevel;

      float t = (float)(adc - lowAdc) / span;

      return lowLevel + t * (highLevel - lowLevel);
    }
  }

  return CAL_LEVEL[CAL_POINTS - 1];
}

/* ===================== 8. RATE OF RISE ================================ */
/* Least-squares slope over the last WIN samples. Steadier than
   (new - old) / dt, which jitters with ADC noise.                      */

void pushSample(float lvl, unsigned long t) {
  levelBuf[bufHead] = lvl; timeBuf[bufHead] = t;
  bufHead = (bufHead + 1) % WIN;
  if (bufCount < WIN) bufCount++;
}

float rateCmPerMin() {
  // Robust Theil-Sen slope over the most recent RATE_WIN samples.
  // This avoids the long tail seen with the previous 60 s least-squares
  // estimator after a rise event has already stopped.
  int n = min(bufCount, RATE_WIN);
  if (n < RATE_WIN) return 0;

  float slopes[(RATE_WIN * (RATE_WIN - 1)) / 2];
  int sc = 0;
  int start = (bufHead - n + WIN) % WIN;

  for (int i = 0; i < n - 1; i++) {
    int idxI = (start + i) % WIN;
    for (int j = i + 1; j < n; j++) {
      int idxJ = (start + j) % WIN;
      float dtMin = (timeBuf[idxJ] - timeBuf[idxI]) / 60000.0f;
      if (dtMin > 0.0f && sc < (int)(sizeof(slopes) / sizeof(slopes[0]))) {
        slopes[sc++] = (levelBuf[idxJ] - levelBuf[idxI]) / dtMin;
      }
    }
  }

  if (sc == 0) return 0;
  for (int i = 1; i < sc; i++) {
    float key = slopes[i];
    int j = i - 1;
    while (j >= 0 && slopes[j] > key) {
      slopes[j + 1] = slopes[j];
      j--;
    }
    slopes[j + 1] = key;
  }

  float r = (sc & 1) ? slopes[sc / 2]
                     : 0.5f * (slopes[sc / 2 - 1] + slopes[sc / 2]);

  if (fabs(r) < RATE_ZERO_BAND) r = 0.0f;
  return constrain(r, -RATE_MAX_CM_MIN, RATE_MAX_CM_MIN);
}

/* ===================== 9. RAIN SENSOR ================================= */

String classifyRain(int adc) {
  if (adc >= RAIN_DRY_ABOVE)   return "Dry";
  if (adc <  RAIN_HEAVY_BELOW) return "Heavy";
  return "Light";
}

int readAvgAdc(int pin) {
  long s = 0;
  for (int i = 0; i < 10; i++) { s += analogRead(pin); delay(2); }
  return (int)(s / 10);
}

/* ===================== 10. DHT11 ======================================= */
/* Reading happens inline in loop() — DHT11 has no meaningful "filter"
   step, only a last-good-value hold when a read momentarily fails.     */

/* ===================== 11. MQTT ======================================== */

void mqttEnsure() {
  if (mqtt.connected() || WiFi.status() != WL_CONNECTED) return;
  static unsigned long last = 0;
  if (millis() - last < 5000) return;      // retry slowly, never block
  last = millis();
  String cid = "esp32-flood-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  bool ok = strlen(MQTT_USER) ? mqtt.connect(cid.c_str(), MQTT_USER, MQTT_PASS)
                              : mqtt.connect(cid.c_str());
  if (ok) {
    mqtt.subscribe(T_PRED);
    mqtt.subscribe(T_CMD);
    Serial.println("MQTT connected");
  } else {
    Serial.print("MQTT retry, rc="); Serial.println(mqtt.state());
  }
}

void wifiEnsure() {
  if (WiFi.status() == WL_CONNECTED) return;
  static unsigned long last = 0;
  if (millis() - last < 10000) return;
  last = millis();
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

// Tiny field readers for the two incoming MQTT payloads — avoids
// pulling in ArduinoJson for three values.
bool jsonNumber(const String& s, const char* key, float& out) {
  int i = s.indexOf(String("\"") + key + "\"");
  if (i < 0) return false;
  i = s.indexOf(':', i);
  if (i < 0) return false;
  out = s.substring(i + 1).toFloat();
  return true;
}
String jsonString(const String& s, const char* key) {
  int i = s.indexOf(String("\"") + key + "\"");
  if (i < 0) return "";
  i = s.indexOf(':', i);            if (i < 0) return "";
  int a = s.indexOf('"', i);        if (a < 0) return "";
  int b = s.indexOf('"', a + 1);    if (b < 0) return "";
  return s.substring(a + 1, b);
}

/* ===================== 12. AI PREDICTION RECEPTION ===================== */
/* The ONLY place this node learns a forecast or a risk level. Nothing
   else in this file is allowed to set lstmLevel / lstmRisk.            */

void onMqtt(char* topic, byte* payload, unsigned int len) {
  String msg; msg.reserve(len);
  for (unsigned int i = 0; i < len; i++) msg += (char)payload[i];
  String t(topic);

  if (t == T_PRED) {                       // genuine forecast from the LSTM
    float p;
    if (jsonNumber(msg, "predicted_level_cm", p)) { lstmLevel = p; lstmAt = millis(); }
    String r = jsonString(msg, "risk");
    if (r.length()) lstmRisk = r;
  }
  else if (t == T_CMD) {                   // pump button on the dashboard
    String m = jsonString(msg, "pump_mode");
    if (m == "auto" || m == "on" || m == "off") {
      pumpMode = m;
      Serial.print("pump mode -> "); Serial.println(pumpMode);
    }
  }
}

/* ===================== 13. PUMP / RELAY CONTROL ======================== */
/* AUTO mode deliberately does NOT run its own risk scoring. It combines
   two things only:
     - the genuine AI risk (HIGH/CRITICAL) received from the laptop
     - a local measured-level safety gate (PUMP_ON_LEVEL_CM /
       PUMP_OFF_LEVEL_CM), so the pump still protects the rig even if
       Wi-Fi/MQTT/the laptop is down — that gate is physical safety,
       not a re-implementation of the removed scoring formula.         */

void writeRelay(bool on) {
  digitalWrite(PIN_RELAY, RELAY_ACTIVE_LOW ? (on ? LOW : HIGH) : (on ? HIGH : LOW));
}

void updatePump() {
  bool want = pumpOn;
  unsigned long now = millis();

  if      (pumpMode == "on")  want = true;
  else if (pumpMode == "off") want = false;
  else {                                                  // auto
    bool aiSaysHigh   = (g_risk == "HIGH" || g_risk == "CRITICAL");
    bool overDangerCm = (g_level >= DANGER_LEVEL_CM);     // local fail-safe
    if (!pumpOn && g_level >= PUMP_ON_LEVEL_CM && (aiSaysHigh || overDangerCm))
      want = true;
    if (pumpOn && g_level <= PUMP_OFF_LEVEL_CM)
      want = false;
  }

  // protection: cap continuous run time, force a rest before restarting
  if (want && pumpOn  && now - pumpChangedAt > PUMP_MAX_RUN_MS)  want = false;
  if (want && !pumpOn && now - pumpChangedAt < PUMP_MIN_REST_MS
           && pumpMode == "auto")                                want = false;

  if (want != pumpOn) {
    pumpOn = want; pumpChangedAt = now; writeRelay(pumpOn);
  }
}

/* ===================== 14. BUZZER ======================================= */
/* The buzzer has two independent meanings:
   1) genuine AI Warning/Danger when that exists, OR
   2) a LOCAL hardware safety alarm based on measured level.
   The local alarm is explicitly NOT an AI risk classifier.
*/

void updateBuzzer() {
  unsigned long now = millis();
  bool localDanger = (g_level >= DANGER_LEVEL_CM);
  bool localWarning = (g_level >= WARN_LEVEL_CM);
  bool aiAlarm = (g_alert == "Warning" || g_alert == "Danger");

  bool shouldBeep = aiAlarm || localWarning;
  bool dangerPattern = localDanger || (g_alert == "Danger");

  if (!shouldBeep) {
    if (buzzState) { buzzState = false; digitalWrite(PIN_BUZZER, LOW); }
    return;
  }

  unsigned long onMs  = dangerPattern ? 400 : 120;
  unsigned long offMs = dangerPattern ? 300  : 2800;
  if (now - tBuzz >= (buzzState ? onMs : offMs)) {
    buzzState = !buzzState;
    digitalWrite(PIN_BUZZER, buzzState ? HIGH : LOW);
    tBuzz = now;
  }
}

/* ===================== 15. JSON GENERATION ============================== */
/* level_cm, rate_cm_min, rain_adc, temp_c, hum are the five feature
   names flood_ai.py's FEATURES list expects verbatim — do not rename
   or remove them. Everything else here is additive.                   */

String buildJson() {
  char b[700];
  snprintf(b, sizeof(b),
    "{\"level_cm\":%.2f,\"rate_cm_min\":%.3f,\"rain_adc\":%d,\"rain_status\":\"%s\","
    "\"water_adc\":%d,\"predicted_cm\":%.2f,\"predict_src\":\"%s\",\"horizon_min\":%d,"
    "\"risk\":\"%s\",\"alert\":\"%s\",\"pump\":\"%s\",\"pump_mode\":\"%s\","
    "\"temp_c\":%.1f,\"hum\":%.1f,\"tank_cm\":%.1f,\"warn_cm\":%.1f,\"danger_cm\":%.1f,"
    "\"sensor_ok\":%s,\"mqtt\":true,\"uptime_s\":%lu}",
    g_level, g_rate, g_rainAdc, g_rain.c_str(),
    g_waterAdc, g_predicted, g_predSrc.c_str(), FORECAST_HORIZON_MIN,
    g_risk.c_str(), g_alert.c_str(), pumpOn ? "ON" : "OFF", pumpMode.c_str(),
    isnan(g_temp) ? 0.0 : g_temp, isnan(g_hum) ? 0.0 : g_hum,
    TANK_CM, WARN_LEVEL_CM, DANGER_LEVEL_CM,
    g_sensorOk ? "true" : "false", millis() / 1000UL);
  return String(b);
}

/* ===================== 16. SETUP ========================================= */

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(PIN_WATER, INPUT);
  pinMode(PIN_RAIN, INPUT);
  pinMode(PIN_BUZZER, OUTPUT); digitalWrite(PIN_BUZZER, LOW);   // buzzer OFF at boot
  pinMode(PIN_RELAY, OUTPUT);  writeRelay(false);               // relay OFF at boot
  analogReadResolution(12);
  measureDryBaseline();
  dht.begin();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Wi-Fi");
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) { delay(250); Serial.print("."); }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("ESP32 IP: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println("No Wi-Fi yet. Sensing, buzzer and pump logic still run locally.");
  }

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqtt);
  mqtt.setBufferSize(768);
}

/* ===================== 17. LOOP ========================================== */

void loop() {
  wifiEnsure();
  mqttEnsure();
  mqtt.loop();

  unsigned long now = millis();

  if (now - tSample >= SAMPLE_MS) {
    tSample = now;

    // ---- water level: burst -> validate -> EMA -> calibrate -> hysteresis ----
    int med, spread;
    sampleWaterAdcRaw(med, spread);
    bool valid = (spread <= ADC_SPREAD_MAX) && (med < ADC_FAULT_HIGH);

    if (med <= DRY_ADC_THRESHOLD) {
      // Fast, explicit dry reset. Do not let EMA create a long false wet tail.
      g_waterAdcEma = 0.0f;
      g_waterAdc = 0;
      g_level = 0.0f;
      g_rate = 0.0f;
      bufCount = 0;
      bufHead = 0;
      g_sensorOk = valid;
    } else {
      if (g_waterAdcEma < 0) g_waterAdcEma = med;
      else g_waterAdcEma = EMA_ALPHA * med + (1 - EMA_ALPHA) * g_waterAdcEma;
      g_waterAdc = (int)(g_waterAdcEma + 0.5f);

      g_sensorOk = valid;
      if (valid) {
        float candidate = constrain(adcToLevelCm(g_waterAdc), 0.0f, TANK_CM);
        if (bufCount == 0 || fabs(candidate - g_level) >= LEVEL_HYST_CM) {
          g_level = candidate;
        }
        pushSample(g_level, now);
      }
    }
    // else: hold the last known g_level. sensor_ok=false is the signal
    // that this ISN'T a confirmed 0 cm dry reading — a genuine fault
    // must never be silently reported as "dry".

    g_rate    = rateCmPerMin();
    g_rainAdc = readAvgAdc(PIN_RAIN);
    g_rain    = classifyRain(g_rainAdc);

    float t = dht.readTemperature(), h = dht.readHumidity();
    if (!isnan(t)) g_temp = t;             // hold last valid reading on failure
    if (!isnan(h)) g_hum  = h;

    // ---- AI reception gate: fresh AI data, or explicitly "unavailable" ----
    if (lstmAt && now - lstmAt < PRED_VALID_MS) {
      g_predicted = lstmLevel;
      g_predSrc   = "lstm";
      g_risk      = lstmRisk.length() ? lstmRisk : "UNKNOWN";
    } else {
      g_predSrc = "unavailable";           // never fabricate a fallback number
      g_risk    = "UNKNOWN";
    }

    if      (g_risk == "CRITICAL")              g_alert = "Danger";
    else if (g_risk == "HIGH")                  g_alert = "Warning";
    else if (g_risk == "LOW" || g_risk == "MODERATE") g_alert = "Normal";
    else                                         g_alert = "Unknown";

    updatePump();

    Serial.print(g_level, 2);
    Serial.print(" cm | ");

    Serial.print(g_rate, 2);
    Serial.print(" cm/min | ");

    Serial.print("rain=");
    Serial.print(g_rain);
    Serial.print(" (adc=");
    Serial.print(g_rainAdc);
    Serial.print(") | ");

    Serial.print("water_adc=");
    Serial.print(g_waterAdc);
    Serial.print(" sensor_ok=");
    Serial.print(g_sensorOk ? "true" : "false");
    Serial.print(" | ");

    Serial.print("temp=");
    if (isnan(g_temp)) Serial.print("NA");
    else Serial.print(g_temp, 1);

    Serial.print(" C hum=");
    if (isnan(g_hum)) Serial.print("NA");
    else Serial.print(g_hum, 1);

    Serial.print("% | ");

    Serial.print("pred ");
    Serial.print(g_predicted, 2);
    Serial.print(" (");
    Serial.print(g_predSrc);
    Serial.print(") | ");

    Serial.print("risk=");
    Serial.print(g_risk);
    Serial.print(" alert=");
    Serial.print(g_alert);
    Serial.print(" | pump ");

    Serial.println(pumpOn ? "ON" : "OFF");
  }

  if (mqtt.connected() && now - tPublish >= PUBLISH_MS) {
    tPublish = now;
    mqtt.publish(T_SENSORS, buildJson().c_str());
  }

  updateBuzzer();
}