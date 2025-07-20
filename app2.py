import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to suppress GUI-related warnings
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # To enable CORS for frontend-backend communication
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import logging

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the static directory and images subdirectory exist
os.makedirs("static/images", exist_ok=True)

# Function to fetch stock data
def fetch_stock_data(stock_symbol, period="2d"):
    try:
        logger.info(f"Fetching data for {stock_symbol} with period {period}")
        data = yf.download(stock_symbol, period=period, interval="1m", auto_adjust=True)
        if data.empty:
            logger.warning(f"No data returned for {stock_symbol}")
            return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Function to predict stock prices
def predict_prices(data):
    try:
        data = data.reset_index()
        data['Timestamp'] = data['Datetime'].apply(lambda x: x.timestamp())
        X = np.array(data['Timestamp']).reshape(-1, 1)
        y = np.array(data['Close']).reshape(-1, 1)

        model = LinearRegression().fit(X, y)
        future_times = [X[-1][0] + 60, X[-1][0] + 3600, X[-1][0] + 86400]  # Next minute, hour, day
        predictions = model.predict(np.array(future_times).reshape(-1, 1))

        # Ensure predictions are not negative
        predictions = np.maximum(predictions, 0)
        
        return predictions.flatten()
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return [0, 0, 0]

# Endpoint to fetch data and generate predictions
@app.route('/get_stock_data', methods=['GET'])
def get_stock_data():
    stock = request.args.get("stock")
    if not stock or stock not in ["Nestle", "MarutiSuzuki", "AxisBank"]:
        logger.warning(f"Invalid stock selected: {stock}")
        return jsonify({"error": "Invalid stock selected. Choose from 'Nestle', 'MarutiSuzuki', or 'AxisBank'."}), 400

    symbols = {"Nestle": "NESTLEIND.NS", "MarutiSuzuki": "MARUTI.NS", "AxisBank": "AXISBANK.NS"}
    stock_symbol = symbols[stock]

    try:
        data = fetch_stock_data(stock_symbol)
        if data.empty:
            return jsonify({"error": "Could not fetch data for the selected stock. This might be due to market hours or API limitations."}), 404

        predictions = predict_prices(data)

        # Plot historical data and predictions
        plt.figure(figsize=(10, 5))
        data['Close'].plot(label="Historical Data")
        future_index = [data.index[-1] + pd.Timedelta(seconds=60),
                        data.index[-1] + pd.Timedelta(hours=1),
                        data.index[-1] + pd.Timedelta(days=1)]
        plt.scatter(future_index, predictions, color='red', label="Predictions")
        plt.title(f"Stock Prices for {stock}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)

        graph_path = f"static/graph_{stock.lower()}.png"
        plt.savefig(graph_path)
        plt.close()
        
        return jsonify({
            "stock": stock,
            "last_price": float(data['Close'].iloc[-1]),
            "predictions": {
                "next_minute": float(predictions[0]),
                "next_hour": float(predictions[1]),
                "next_day": float(predictions[2])
            },
            "graph_url": f"/{graph_path}"
        })
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Serve index2.html (Homepage)
@app.route('/')
def serve_index2():
    return send_from_directory('.', 'index2.html')

# Serve stock2.html (Stock Page)
@app.route('/stock2.html')
def serve_stock2():
    return send_from_directory('.', 'stock2.html')

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Handle favicon requests
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content response to suppress favicon errors

if __name__ == '__main__':
    app.run(debug=True)