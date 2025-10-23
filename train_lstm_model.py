import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import matplotlib.pyplot as plt

# ===========================
# LOAD AND PREPARE THE DATA
# ===========================

print("📥 Loading and processing tick data...")
df = pd.read_csv('tick_data.csv')
df.rename(columns={df.columns[1]: 'Price'}, inplace=True)

# === Calculate Technical Indicators ===
print("📊 Calculating EMA, RSI, MACD...")

# EMA (Exponential Moving Average)
df['EMA'] = df['Price'].ewm(span=14, adjust=False).mean()

# RSI (Relative Strength Index)
delta = df['Price'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
avg_gain = up.rolling(14).mean()
avg_loss = down.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD and Signal Line
ema_fast = df['Price'].ewm(span=12, adjust=False).mean()
ema_slow = df['Price'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_fast - ema_slow
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Drop any rows with NaN (from rolling calculations)
df.dropna(inplace=True)

# ===========================
# SCALING AND SEQUENCING
# ===========================

print("⚙️ Scaling features and preparing sequences...")

feature_cols = ['Price', 'EMA', 'RSI', 'MACD', 'MACD_signal']
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

# Scale using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_df[feature_cols])
train_scaled = scaler.transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
joblib.dump(scaler, 'scaler.pkl')

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i][0])  # Predicting next price
    return np.array(X), np.array(y)

seq_len = 60
X_train, y_train = create_sequences(train_scaled, seq_len)

# Append last 60 points of training to front of validation set
val_with_context = np.concatenate([train_scaled[-seq_len:], val_scaled], axis=0)
X_val, y_val = create_sequences(val_with_context, seq_len)

# ===========================
# BUILD LSTM MODEL
# ===========================

print("🧠 Building LSTM model...")
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(seq_len, len(feature_cols))))  # Increased units
model.add(Dropout(0.3))  # Increased dropout
model.add(LSTM(64, return_sequences=False))  # Added more complexity
model.add(Dropout(0.3))
model.add(Dense(1))  # Output: predicted next price
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# ===========================
# TRAIN THE MODEL
# ===========================

print("🚀 Training the model...")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)  # Increased patience
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)  # Save in .keras format

# Learning rate scheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint, reduce_lr],  # Added ReduceLROnPlateau
    shuffle=True
)

# ===========================
# PLOT TRAINING LOSS
# ===========================

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.savefig("charts/training_curve.png")
plt.show()

# ===========================
# SAVE FINAL MODEL
# ===========================

print("💾 Saving model and scaler...")
model.save("final_model.keras")  # Save in .keras format

# ===========================
# EVALUATE SIGNAL ACCURACY
# ===========================

print("📈 Generating buy/sell signals on validation set...")

y_pred = model.predict(X_val)

signals = []
for i in range(len(y_pred)):
    current_price = X_val[i, -1, 0]
    predicted_price = y_pred[i, 0]
    signals.append("Buy" if predicted_price > current_price else "Sell")

actual_movement = []
for i in range(len(y_val)):
    actual_price = y_val[i]
    current_price = X_val[i, -1, 0]
    actual_movement.append("Buy" if actual_price > current_price else "Sell")

accuracy = sum([signals[i] == actual_movement[i] for i in range(len(signals))]) / len(signals)
print(f"✅ Signal prediction accuracy: {accuracy:.2%}")
