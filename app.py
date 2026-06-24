from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import json
import requests
from math import isnan


app = Flask(__name__)

# Load NEW retrained model & encoders
model = joblib.load("crop_top3_model_new.joblib")
label_encoder = joblib.load("crop_label_encoder_new.joblib")
soil_texture_encoder = joblib.load("soil_texture_encoder.joblib")

# Load NEW feature columns
with open("feature_columns_new.json", "r") as f:
    feature_order = json.load(f)

# HTML FORM (UI SAME AS ORIGINAL + NEW SOIL MOISTURE FIELD)
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Crop Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #e8f0f7, #cfd9df);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }

        .container {
            background: white;
            padding: 25px 35px;
            border-radius: 14px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
            width: 450px;
            animation: fadeIn 0.6s ease;
        }

        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.9em;
            letter-spacing: 0.5px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px 18px;
        }

        .full-width {
            grid-column: 1 / span 2;
        }

        label {
            font-weight: 600;
            color: #34495e;
            font-size: 14px;
            display: block;
            margin-bottom: 4px;
        }

        input[type="number"], select {
            width: 100%;
            padding: 8px 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 14px;
            background: #fafafa;
            transition: all 0.3s ease;
        }

        input[type="number"]:focus, select:focus {
            border-color: #27ae60;
            background: #fff;
            outline: none;
            box-shadow: 0 0 5px rgba(39, 174, 96, 0.3);
        }

        select {
            cursor: pointer;
        }

        input[type="submit"] {
            background: #27ae60;
            color: white;
            border: none;
            padding: 11px;
            margin-top: 18px;
            width: 100%;
            border-radius: 6px;
            font-size: 15px;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.2s ease;
        }

        input[type="submit"]:hover {
            background: #219150;
            transform: translateY(-1px);
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
</head>
<body onload="autofillAll(); setTimeout(autofillNPK_from_server, 800);">



    <div class="container">
        <h1>Crop Prediction</h1>
        <form action="/predict" method="post" class="form-grid">
            
            <div>
                <label for="N">N:</label>
                <input type="number" name="N" step="any">
            </div>

            <div>
                <label for="P">P:</label>
                <input type="number" name="P" step="any">
            </div>

            <div>
                <label for="K">K:</label>
                <input type="number" name="K" step="any">
            </div>

            <div>
                <label for="temperature">Temperature:</label>
                <input type="number" name="temperature" step="any">
            </div>

            <div class="full-width">
                <label for="humidity">Humidity:</label>
                <input type="number" name="humidity" step="any">
            </div>

            <div class="full-width">
                <label for="ph">pH:</label>
                <input type="number" name="ph" step="any">
            </div>

            <div class="full-width">
                <label for="rainfall">Rainfall:</label>
                <input type="number" name="rainfall" step="any">
            </div>

            <div class="full-width">
                <label for="organic_matter_content">Organic Matter Content:</label>
                <input type="number" name="organic_matter_content" step="any">
            </div>

            <div class="full-width">
                <label for="crop_cycle_duration">Crop Cycle Duration:</label>
                <input type="number" name="crop_cycle_duration" step="any">
            </div>

            <div class="full-width">
                <label for="soil_texture">Soil Texture Type:</label>
                <select name="soil_texture" id="soil_texture" required>
                    <option value="" disabled selected>Select soil texture</option>
                    <option value="sandy">Sandy</option>
                    <option value="loamy">Loamy</option>
                    <option value="clayey">Clay</option>
                    <option value="silty">Silty</option>
                </select>
            </div>

            <div class="full-width">
                <label for="soil_moisture">Soil Moisture (%):</label>
                <input type="number" name="soil_moisture" step="any">
            </div>

            <div class="full-width">
                <input type="submit" value="Predict">
            </div>
        </form>
    </div>
    <script>
// ================================
// AUTO-FILL ALL WEATHER VARIABLES
// ================================
async function autofillAll() {

    if (!navigator.geolocation) {
        alert("Location not supported");
        return;
    }

    navigator.geolocation.getCurrentPosition(async function(pos) {

        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;

        console.log("GPS:", lat, lon);

        // ------------ 1. TEMPERATURE & HUMIDITY (Last 7 days avg) ------------
        const end = new Date().toISOString().split("T")[0];
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - 7);
        const start = startDate.toISOString().split("T")[0];

        const weatherURL =
            `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}` +
            `&start_date=${start}&end_date=${end}&daily=temperature_2m_mean,relative_humidity_2m_mean&timezone=auto`;

        try {
            const resWeather = await fetch(weatherURL);
            const dataWeather = await resWeather.json();

            // Avg temp
            const temps = dataWeather.daily?.temperature_2m_mean || [];
            let avgTemp = temps.reduce((a, b) => a + (b || 0), 0) / temps.length;

            // Avg humidity
            const hums = dataWeather.daily?.relative_humidity_2m_mean || [];
            let avgHum = hums.reduce((a, b) => a + (b || 0), 0) / hums.length;

            document.querySelector('input[name="temperature"]').value =
                avgTemp.toFixed(2);

            document.querySelector('input[name="humidity"]').value =
                avgHum.toFixed(2);

            console.log("Auto Temp:", avgTemp);
            console.log("Auto Humidity:", avgHum);

        } catch (error) {
            console.error("Temp/Humidity fetch error:", error);
        }

        // ------------ 2. RAINFALL (last 60 days) ------------
        const rainStartDate = new Date();
        rainStartDate.setDate(rainStartDate.getDate() - 60);
        const rainStart = rainStartDate.toISOString().split("T")[0];

        const rainURL =
            `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}` +
            `&start_date=${rainStart}&end_date=${end}&daily=precipitation_sum&timezone=auto`;

        try {
            const resRain = await fetch(rainURL);
            const dataRain = await resRain.json();

            const rainfallData = dataRain.daily?.precipitation_sum || [];
            let totalRain = rainfallData.reduce((a, b) => a + (b || 0), 0);

            document.querySelector('input[name="rainfall"]').value =
                totalRain.toFixed(2);

            console.log("Auto Rainfall:", totalRain);

        } catch (error) {
            console.error("Rainfall error:", error);
        }

        // ------------ 3. SOIL MOISTURE (NASA SMAP API) ------------
        const soilURL =
            `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
            `&hourly=soil_moisture_0_1cm`;

        try {
            const resSoil = await fetch(soilURL);
            const dataSoil = await resSoil.json();

            const soilData = dataSoil.hourly?.soil_moisture_0_1cm || [];

            let soilMoist = soilData[0] ? soilData[0] * 100 : 20; // convert to %

            document.querySelector('input[name="soil_moisture"]').value =
                soilMoist.toFixed(2);

            console.log("Auto Soil Moisture:", soilMoist);

        } catch (error) {
            console.error("Soil moisture error:", error);
        }
        // ------------ 4. ORGANIC MATTER CONTENT (SoilGrids SOC API) ------------
try {
    const socURL =
        `https://rest.isric.org/soilgrids/v2.0/properties/query?lat=${lat}&lon=${lon}&property=soc`;

    const resSOC = await fetch(socURL);
    const dataSOC = await resSOC.json();

    let soc_gkg = null;

    // Extract SOC -> g/kg
    try {
        const props = dataSOC.properties || {};
        const socData = props.soc?.values?.[0];

        if (socData?.value !== undefined) {
            soc_gkg = socData.value;  // g/kg
        }
    } catch (e) {
        console.log("SOC extraction error:", e);
    }

    // fallback
    if (!soc_gkg) soc_gkg = 10; // typical SOC value

    // SOC g/kg → SOC % (divide by 10)
    const soc_percent = soc_gkg / 10.0;

    // SOC% → Organic Matter % (Van Bemmelen factor)
    const organic_matter = soc_percent * 1.724;

    // Fill in UI
    document.querySelector('input[name="organic_matter_content"]').value =
        organic_matter.toFixed(2);

    console.log("Auto Organic Matter Content (%):", organic_matter);

} catch (error) {
    console.error("Organic Matter fetch error:", error);
}

    });
}
async function autofillNPK_from_server() {
    if (!navigator.geolocation) {
        alert("Geolocation not supported");
        return;
    }

    navigator.geolocation.getCurrentPosition(async function(pos) {

        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;

        try {
            const resp = await fetch("/autofill_npk", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lat: lat, lon: lon })
            });

            if (!resp.ok) {
                console.error("NPK endpoint error", await resp.text());
                return;
            }

            const data = await resp.json();

            // Fill inputs
            // Fill inputs
// Fill inputs
if (data.N !== undefined)
    document.querySelector('input[name="N"]').value = data.N;

if (data.P !== undefined)
    document.querySelector('input[name="P"]').value = data.P;

if (data.K !== undefined)
    document.querySelector('input[name="K"]').value = data.K;

// Organic Matter autofill
if (data.organic_matter !== undefined)
    document.querySelector('input[name="organic_matter_content"]').value = data.organic_matter;

// ⭐ Soil pH autofill
if (data.ph !== undefined)
    document.querySelector('input[name="ph"]').value = data.ph;

console.log("Autofilled NPK + OM + pH:", data);



            // If you want to inspect debug values:
            // console.log(data.debug);

        } catch (err) {
            console.error("autofillNPK error", err);
        }
    });
}
</script>







</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_form)

@app.route('/predict', methods=['POST'])
def predict():
    # Read form data
    data = request.form.to_dict()

    # Convert numeric fields
    for k in data:
        if k not in ["soil_texture"]:
            data[k] = float(data[k])

    # Encode soil texture
    data["soil_texture_encoded"] = float(
        soil_texture_encoder.transform([data["soil_texture"]])[0]
    )
    del data["soil_texture"]

    # Create DataFrame in correct order
    df = pd.DataFrame([data], columns=feature_order)

    # Predict top 3 crops
    proba = model.predict_proba(df)[0]
    top3_idx = proba.argsort()[-3:][::-1]
    crops = label_encoder.inverse_transform(top3_idx)

    result = [
        {"crop": crops[i], "probability": float(proba[top3_idx[i]])}
        for i in range(3)
    ]
    return jsonify(result)

@app.route('/autofill_npk', methods=['POST'])
def autofill_npk():
    """
    Expects JSON: {"lat": <float>, "lon": <float>}
    Returns estimated N, P, K as JSON.
    """
    try:
        payload = request.get_json(force=True)
        lat = float(payload.get("lat"))
        lon = float(payload.get("lon"))
    except Exception as e:
        return jsonify({"error": "invalid payload", "detail": str(e)}), 400

    # --- 1) Get soil properties from SoilGrids (ISRIC) ---
    # We will request clay, sand, soc (soil organic carbon), phh2o
    soilgrids_url = (
        f"https://rest.isric.org/soilgrids/v2.0/properties/query"
        f"?lat={lat}&lon={lon}&property=clay&property=sand&property=soc&property=phh2o"
    )

    clay_pct = None
    sand_pct = None
    soc_gkg = None   # g/kg
    soil_ph = None

    try:
        r = requests.get(soilgrids_url, timeout=15)
        r.raise_for_status()
        sg = r.json()

        # SoilGrids responses can be nested. We'll try multiple safe paths.
        props = sg.get("properties", {})

        def extract_first_value(props, key):
            # try several possible structures
            v = None
            if key in props:
                # common structure: props[key]['values'][0]['value']
                try:
                    values = props[key].get("values")
                    if isinstance(values, list) and len(values):
                        # some props have list of dicts with 'value'
                        first = values[0]
                        # some keys: first may have 'depths' or 'value'
                        if isinstance(first, dict):
                            if "value" in first:
                                return first.get("value")
                            # some responses have 'depths' list
                            if "depths" in first and isinstance(first["depths"], list):
                                d0 = first["depths"][0]
                                if isinstance(d0, dict) and "value" in d0:
                                    return d0["value"]
                except Exception:
                    pass

            # fallback: search entire JSON for key substrings
            def deep_search(obj, target):
                if isinstance(obj, dict):
                    for k, val in obj.items():
                        if k.lower() == target.lower():
                            return val
                        res = deep_search(val, target)
                        if res is not None:
                            return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = deep_search(item, target)
                        if res is not None:
                            return res
                return None

            found = deep_search(props, key)
            # if found is numeric or dict with 'value'
            if isinstance(found, (int, float)):
                return found
            if isinstance(found, dict) and "value" in found:
                return found["value"]
            return None

        clay_pct = extract_first_value(props, "clay")
        sand_pct = extract_first_value(props, "sand")
        soc_gkg = extract_first_value(props, "soc")
        soil_ph = extract_first_value(props, "phh2o")

    except Exception as e:
        # If SoilGrids fails, we continue with None values and use fallbacks
        print("SoilGrids fetch error:", e)

    # --- 2) Get recent soil moisture from Open-Meteo (hourly shallow soil moisture) ---
    soil_moisture_percent = None
    try:
        om_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=soil_moisture_0_1cm&timezone=auto"
        r2 = requests.get(om_url, timeout=12)
        r2.raise_for_status()
        om = r2.json()
        sm = om.get("hourly", {}).get("soil_moisture_0_1cm", [])
        if sm and len(sm) > 0:
            # take the latest non-null value and convert to percent (if units are m3/m3; here we assume m3/m3)
            latest = None
            # find first numeric
            for v in sm:
                if v is not None:
                    latest = v
                    break
            if latest is not None:
                # commonly soil_moisture reported in m3/m3 (0-0.5). Convert to %.
                soil_moisture_percent = float(latest) * 100.0
    except Exception as e:
        print("Open-Meteo soil moisture fetch error:", e)

    # --- 3) Fallback defaults (if any values missing) ---
    if clay_pct is None:
        clay_pct = 20.0  # assume light loam
    if sand_pct is None:
        sand_pct = 40.0
    if soc_gkg is None:
        soc_gkg = 10.0  # 10 g/kg = 1% organic carbon
    if soil_ph is None:
        soil_ph = 6.5
    if soil_moisture_percent is None:
        soil_moisture_percent = 25.0

    # --- 4) Convert/normalize soil properties ---
    try:
        clay_pct = float(clay_pct)
    except:
        clay_pct = 20.0
    try:
        sand_pct = float(sand_pct)
    except:
        sand_pct = 40.0
    try:
        soc_gkg = float(soc_gkg)   # g/kg
    except:
        soc_gkg = 10.0
    try:
        soil_ph = float(soil_ph)
    except:
        soil_ph = 6.5
    try:
        soil_moisture_percent = float(soil_moisture_percent)
    except:
        soil_moisture_percent = 25.0

    # Convert SOC g/kg -> % (g/kg divided by 10)
    soc_pct = soc_gkg / 10.0   # e.g., 10 g/kg -> 1.0%

    # Estimate Organic Matter (%) from SOC: OM ≈ SOC * 1.724
    organic_matter_pct = soc_pct * 1.724

    # --- 5) Empirical formulas to estimate N, P, K ---
    # NOTE: these formulas are empirical approximations designed to produce values
    # on the same scale as typical crop datasets (N,P,K values ~ 20-300).
    # Feel free to tune the constants below with local soil-test data.

    # Nitrogen (N) - correlated with organic matter and soil moisture
    # base_N = organic_matter_pct * 30   (approx scale)
    moisture_factor = 1.0 + (soil_moisture_percent - 25.0) / 200.0
    moisture_factor = max(0.6, min(1.6, moisture_factor))
    base_N = organic_matter_pct * 30.0
    estimated_N = base_N * moisture_factor

    # Phosphorus (P) - depends on soil texture and pH (availability peaks near pH ~6.5)
    clay_factor = 1.0 + (clay_pct - 20.0) / 200.0   # clay increases P holding capacity a bit
    ph_factor = 1.0 - (abs(soil_ph - 6.5) / 5.0)    # reduces availability if pH far from 6.5
    ph_factor = max(0.5, min(1.2, ph_factor))
    base_P = 15.0 + clay_pct * 0.08
    estimated_P = base_P * clay_factor * ph_factor

    # Potassium (K) - strongly correlated with clay% and organic matter
    estimated_K = 40.0 + clay_pct * 1.0 + organic_matter_pct * 12.0

    # Clamp and round to sensible ranges (dataset-like)
    def clamp(x, a, b):
        return max(a, min(b, x))

    N_final = round(clamp(estimated_N, 5.0, 300.0), 2)
    P_final = round(clamp(estimated_P, 2.0, 200.0), 2)
    K_final = round(clamp(estimated_K, 10.0, 500.0), 2)

    result = {
        "N": N_final,
        "P": P_final,
        "K": K_final,
        "organic_matter": round(organic_matter_pct, 2),
        "ph": round(soil_ph, 2),
        "debug": {
            "clay_pct": clay_pct,
            "sand_pct": sand_pct,
            "soc_gkg": soc_gkg,
            "soc_pct": round(soc_pct,3),
            "organic_matter_pct": round(organic_matter_pct,3),
            "soil_ph": round(soil_ph,3),
            "soil_moisture_percent": round(soil_moisture_percent,3),
            "estimated_N_raw": round(estimated_N,3),
            "estimated_P_raw": round(estimated_P,3),
            "estimated_K_raw": round(estimated_K,3)
        }
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
