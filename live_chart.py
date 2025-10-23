# === live_chart_5s.py ===

import sqlite3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import joblib
import time

model = load_model("best_model.h5")
scaler = joblib.load("scaler.pkl")
conn = sqlite3.connect('market_data.db', check_same_thread=False)
cursor = conn.cursor()

fig, ax = plt.subplots()
predicted_prices = deque(maxlen=100)
predicted_timestamps = deque(maxlen=100)
buy_signals = deque(maxlen=100)
sell_signals = deque(maxlen=100)

# Config
seq_len = 60
future_steps = 5  # simulate 5s-ahead

# Signal control
last_signal_time = 0
min_interval = 5.0
current_position = None

# File logging
def write_signal_to_file(signal, timestamp, price):
    with open("signals.txt", "a") as file:
        file.write(f"{signal},{timestamp},{price}\n")

# Animation loop
def animate(i):
    global last_signal_time, current_position

    cursor.execute("""
        SELECT timestamp, price FROM ticks
        WHERE symbol = 'AUDCAD_otc'
        ORDER BY timestamp DESC LIMIT 200
    """)
    rows = cursor.fetchall()[::-1]
    if len(rows) < seq_len + future_steps:
        return

    timestamps = [r[0] for r in rows]
    prices = [r[1] for r in rows]

    df = pd.DataFrame({'price': prices})
    df['EMA'] = df['price'].ewm(span=14, adjust=False).mean()
    delta = df['price'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(14).mean()
    avg_loss = down.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema_fast = df['price'].ewm(span=12, adjust=False).mean()
    ema_slow = df['price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df.dropna(inplace=True)
    if len(df) < seq_len:
        return

    last_seq = df[['price', 'EMA', 'RSI', 'MACD', 'MACD_signal']].values[-seq_len:]
    last_seq_scaled = scaler.transform(last_seq)
    input_seq = np.reshape(last_seq_scaled, (1, seq_len, 5))

    # Forecast 5 steps forward by rolling prediction
    next_pred = None
    pred_seq = last_seq_scaled.copy()
    for _ in range(future_steps):
        step_input = np.reshape(pred_seq[-seq_len:], (1, seq_len, 5))
        step_pred_scaled = model.predict(step_input, verbose=0)
        new_step = pred_seq[-1].copy()
        new_step[0] = step_pred_scaled[0][0]
        pred_seq = np.vstack([pred_seq, new_step])
        next_pred = new_step

    future_price_scaled = next_pred
    predicted_price = scaler.inverse_transform([future_price_scaled])[0][0]

    tick_interval = timestamps[-1] - timestamps[-2]
    next_timestamp = timestamps[-1] + future_steps * tick_interval
    predicted_prices.append(predicted_price)
    predicted_timestamps.append(next_timestamp)

    # Improved signal logic
    current_price = prices[-1]
    recent_changes = np.diff(prices[-15:])
    volatility = pd.Series(recent_changes).ewm(span=10, adjust=False).mean().abs().mean()
    min_threshold = 0.0001
    threshold = max(volatility * 0.7, min_threshold)

    signal = None
    if predicted_price > current_price + threshold:
        signal = "BUY"
    elif predicted_price < current_price - threshold:
        signal = "SELL"

    now = time.time()
    if signal and now - last_signal_time >= min_interval:
        if signal == "BUY":
            buy_signals.append((next_timestamp, predicted_price))
            sell_signals.append((next_timestamp, np.nan))
        elif signal == "SELL":
            sell_signals.append((next_timestamp, predicted_price))
            buy_signals.append((next_timestamp, np.nan))
        write_signal_to_file(signal, next_timestamp, predicted_price)
        last_signal_time = now
        current_position = signal
    else:
        buy_signals.append((next_timestamp, np.nan))
        sell_signals.append((next_timestamp, np.nan))

    # Plotting
    ax.clear()
    ax.plot(timestamps[-100:], prices[-100:], label='Actual Price', color='blue')
    ax.plot(predicted_timestamps, predicted_prices, label='Predicted 5s Ahead', color='red')

    if buy_signals:
        buy_t, buy_p = zip(*buy_signals)
        ax.scatter(buy_t, buy_p, marker='^', color='green', label='Buy')
    if sell_signals:
        sell_t, sell_p = zip(*sell_signals)
        ax.scatter(sell_t, sell_p, marker='v', color='orange', label='Sell')

    # Dynamic Y-axis
    all_prices = np.array(prices[-100:] + list(predicted_prices))
    y_min, y_max = all_prices.min(), all_prices.max()
    y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Auto-focus on latest data
    if len(timestamps) > 100:
        ax.set_xlim(timestamps[-100], predicted_timestamps[-1])

    ax.set_title("AUDCAD_otc - Live vs 5s Ahead Prediction")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.tight_layout()
plt.show()
