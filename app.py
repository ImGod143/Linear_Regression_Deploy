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

# Embedded HTML Template with Modern Slate & Emerald Color Combination
HTML_TEMPLATE = """
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
            border: 1.5px solid
