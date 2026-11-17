import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Linear Regression model safely using absolute path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Embedded HTML Template
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Property Value Predictor</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: #ffffff;
            --primary: #10b981;
            --primary-hover: #059669;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border-color: #cbd5e1;
            --input-focus: #10b981;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }
        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: var(--text-main);
        }
        .container {
            background: var(--card-bg);
            max-width: 650px;
            width: 100%;
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }
        .header {
            background: #0f172a;
            color: #ffffff;
            padding: 28px 32px;
            text-align: center;
            border-bottom: 4px solid var(--primary);
        }
        .header h1 {
            font-size: 1.65rem;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: -0.025em;
        }
        .header p {
            color: #94a3b8;
            font-size: 0.95rem;
        }
        .form-body {
            padding: 32px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 6px;
        }
        input {
            padding: 12px 14px;
            border: 1.5px solid var(--border-color);
            border-radius: 8px;
            font-size: 1rem;
            color: var(--text-main);
            transition: all 0.2s ease;
            outline: none;
            background: #f8fafc;
        }
        input:focus {
            border-color: var(--input-focus);
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15);
        }
        .btn-submit {
            margin-top: 28px;
            width: 100%;
            padding: 14px;
            background: var(--primary);
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.1s ease;
        }
        .btn-submit:hover {
            background: var(--primary-hover);
        }
        .btn-submit:active {
            transform: scale(0.99);
        }
        .result-box {
            margin-top: 24px;
            padding: 20px;
            border-radius: 10px;
            background: #ecfdf5;
            border: 1.5px solid #6ee7b7;
            text-align: center;
        }
        .result-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #047857;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .result-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #065f46;
        }
        .error-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 10px;
            background: #fef2f2;
            border: 1.5px solid #fca5a5;
            color: #991b1b;
            text-align: center;
            font-weight: 500;
        }
        @media (max-width: 550px) {
            .grid {
                grid-template-columns: 1fr;
            }
            .form-group.full-width {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Real Estate Value Predictor</h1>
            <p>Enter property details below to estimate market value</p>
        </div>
        <div class="form-body">
            <form method="POST" action="/">
                <div class="grid">
                    <div class="form-group">
                        <label for="Square_Footage">Square Footage</label>
                        <input type="number" step="any" name="Square_Footage" id="Square_Footage" placeholder="e.g. 2500" value="{{ form_data.get('Square_Footage', '') }}" required>
                    </div>
                    <div class="form-group">
                        <label for="Num_Bedrooms">Number of Bedrooms</label>
                        <input type="number" step="1" name="Num_Bedrooms" id="Num_Bedrooms" placeholder="e.g. 3" value="{{ form_data.get('Num_Bedrooms', '') }}" required>
                    </div>
                    <div class="form-group">
                        <label for="Num_Bathrooms">Number of Bathrooms</label>
                        <input type="number" step="any" name="Num_Bathrooms" id="Num_Bathrooms" placeholder="e.g. 2" value="{{ form_data.get('Num_Bathrooms', '') }}" required>
                    </div>
                    <div class="form-group">
                        <label for="Year_Built">Year Built</label>
                        <input type="number" step="1" name="Year_Built" id="Year_Built" placeholder="e.g. 2010" value="{{ form_data.get('Year_Built', '') }}" required>
                    </div>
                    <div class="form-group">
                        <label for="Lot_Size">Lot Size (Acres / Sq Ft)</label>
                        <input type="number" step="any" name="Lot_Size" id="Lot_Size" placeholder="e.g. 1.5" value="{{ form_data.get('Lot_Size', '') }}" required>
                    </div>
                    <div class="form-group">
                        <label for="Garage_Size">Garage Size (Cars)</label>
                        <input type="number" step="1" name="Garage_Size" id="Garage_Size" placeholder="e.g. 2" value="{{ form_data.get('Garage_Size', '') }}" required>
                    </div>
                    <div class="form-group full-width">
                        <label for="Neighborhood_Quality">Neighborhood Quality (Rating 1 - 10)</label>
                        <input type="number" step="1" min="1" max="10" name="Neighborhood_Quality" id="Neighborhood_Quality" placeholder="e.g. 8" value="{{ form_data.get('Neighborhood_Quality', '') }}" required>
                    </div>
                </div>
                <button type="submit" class="btn-submit">Predict Property Value</button>
            </form>

            {% if prediction is not none %}
            <div class="result-box">
                <div class="result-label">Estimated Value</div>
                <div class="result-value">${{ "{:,.2f}".format(prediction) }}</div>
            </div>
            {% endif %}

            {% if error %}
            <div class="error-box">
                {{ error }}
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    form_data = {}

    if request.method == 'POST':
        try:
            form_data = request.form
            features = pd.DataFrame([{
                'Square_Footage': float(request.form['Square_Footage']),
                'Num_Bedrooms': float(request.form['Num_Bedrooms']),
                'Num_Bathrooms': float(request.form['Num_Bathrooms']),
                'Year_Built': float(request.form['Year_Built']),
                'Lot_Size': float(request.form['Lot_Size']),
                'Garage_Size': float(request.form['Garage_Size']),
                'Neighborhood_Quality': float(request.form['Neighborhood_Quality'])
            }])
            
            pred_value = model.predict(features)[0]
            prediction = float(pred_value)
        except Exception as e:
            error = f"Prediction Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error, form_data=form_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
