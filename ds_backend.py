import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# Define the list of stocks and date range
stocks = ['GOOG', 'AAPL', 'MSFT', 'AMZN']
start = '2012-01-01'
end = '2022-12-21'

# Download and preprocess data for each stock
data = {}
for stock in stocks:
    data[stock] = yf.download(stock, start, end)
    data[stock].reset_index(inplace=True)

# Train and predict for each stock
for stock in stocks:
    print(f"Training and predicting for stock: {stock}")
    
    # Calculate 100-day moving average
    ma_100_days = data[stock].Close.rolling(100).mean()
    
    # Split data into training and testing sets
    data_train = pd.DataFrame(data[stock].Close[0: int(len(data[stock])*0.80)])
    data_test = pd.DataFrame(data[stock].Close[int(len(data[stock])*0.80): len(data[stock])])
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scale = scaler.fit_transform(data_train)
    
    # Prepare data for LSTM model
    x_train, y_train = [], []
    for i in range(100, data_train_scale.shape[0]):
        x_train.append(data_train_scale[i-100:i])
        y_train.append(data_train_scale[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the LSTM model
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Prepare test data
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)
    
    x_test, y_test = [], []
    for i in range(100, data_test_scale.shape[0]):
        x_test.append(data_test_scale[i-100:i])
        y_test.append(data_test_scale[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # Predict using the LSTM model
    y_predict = model.predict(x_test)
    scale = 1 / scaler.scale_
    y_predict = y_predict * scale
    y_test = y_test * scale
    
    # Plot the predictions and original prices for each stock
    plt.figure(figsize=(10, 8))
    plt.plot(y_predict, 'r', label='Predicted Price')
    plt.plot(y_test, 'g', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f"Predicted vs Original Price for {stock}")
    plt.legend()
    plt.show()
    
    # Save the model
    model.save(f'Stock_Predictions_Model_{stock}.keras')


    from flask import Flask, jsonify, render_template,request
import pickle
import numpy as np
import yfinance as yf

app = Flask(__name__)
model = pickle.load(open('Stock_Predictions_Model_GOOG.keras', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    stock = request.args.get('stock')
    data = yf.download(stock, start='2012-01-01', end='2022-12-21')
    data = data['Close'].values.reshape(-1, 1)
    data = scaler.transform(data)
    x_test = []
    for i in range(100, data.shape[0]):
        x_test.append(data[i-100:i])
    x_test = np.array(x_test)
    y_test = model.predict(x_test)
    y_test = scaler.inverse_transform(y_test)
    prices = data[100:, 0]
    prices = scaler.inverse_transform(prices.reshape(-1, 1))

    return jsonify({
        'prices': prices.tolist(),
        'predicted_prices': y_test.tolist()
    })

if __name__ == '_main_':
    app.run(debug=True)