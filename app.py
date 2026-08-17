from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler objects safely
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error_text="Model file 'model.pkl' not found.")
    if scaler is None:
        return render_template('index.html', error_text="Scaler file 'scaler.pkl' not found.")

    try:
        # 1. Extract inputs from the web form
        lat = float(request.form.get('latitude', 0))
        lon = float(request.form.get('longitude', 0))
        station_count = float(request.form.get('station_count', 1))
        port_count = float(request.form.get('port_count', 1))
        country_code = request.form.get('country', 'Other')

        # 2. Re-engineer the exact notebook features
        ports_per_station = port_count / station_count if station_count > 0 else 0

        # 3. Map HTML short codes to the exact full names seen at training time
        code_to_name = {
            'US': 'United States',
            'DE': 'Germany',
            'FR': 'France',
            'GB': 'United Kingdom',
            'NO': 'Norway',
            'SE': 'Sweden',
            'CA': 'Canada',
            'NL': 'Netherlands',
            'AU': 'Australia',
            'BE': 'Belgium',
            'IT': 'Italy',
            'PT': 'Portugal',
            'ES': 'Spain'
        }
        
        full_country_name = code_to_name.get(country_code, 'Other')
        active_country_column = f"country_group_{full_country_name}"

        # 4. Dynamically get the EXACT columns the model/scaler expects!
        # This completely prevents any "feature shape mismatch" errors on AWS
        expected_order = list(scaler.feature_names_in_)

        # Initialize base payload DataFrame with zeros using the exact expected columns
        input_df = pd.DataFrame(0.0, index=[0], columns=expected_order)
        
        # Populate continuous numerical variables (checking to ensure column exists)
        if 'latitude' in input_df.columns: 
            input_df['latitude'] = lat
        if 'longitude' in input_df.columns: 
            input_df['longitude'] = lon
        if 'station_count' in input_df.columns: 
            input_df['station_count'] = station_count
        if 'port_count' in input_df.columns: 
            input_df['port_count'] = port_count
        if 'ports_per_station' in input_df.columns: 
            input_df['ports_per_station'] = ports_per_station
        
        # Set active one-hot country column to 1.0; fall back to 'Other' if not matched
        if active_country_column in input_df.columns:
            input_df[active_country_column] = 1.0
        elif 'country_group_Other' in input_df.columns:
            input_df['country_group_Other'] = 1.0

        # 5. Apply the scaling transform across the complete structured DataFrame
        scaled_features = scaler.transform(input_df)

        # 6. Generate non-trivial classification prediction
        prediction = model.predict(scaled_features)
        is_fast_charging = int(prediction[0]) == 1

        if is_fast_charging:
            result_status = "success"
            result_text = "⚡ High-Speed DC Fast Charging Available"
            result_desc = "This location matches the physical footprint and regional infrastructure signatures of a high-power charging hub."
        else:
            result_status = "standard"
            result_text = "🔌 Standard AC / Low-Power Charging Only"
            result_desc = "Based on infrastructure layout footprints, this location is highly predicted to offer standard charging speeds only."

        return render_template(
            'index.html', 
            prediction_text=result_text, 
            prediction_status=result_status,
            prediction_desc=result_desc,
            prev_lat=lat, prev_lon=lon, prev_stations=int(station_count), prev_ports=int(port_count), prev_country=country_code
        )

    except Exception as e:
        return render_template('index.html', error_text=f"Processing Error: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
