import time
import subprocess

# Train every 30 minutes (1800 seconds)
TRAIN_INTERVAL = 600  # 30 minutes

print("🔁 Auto-trainer started...")

while True:
    print("🚀 Starting model training...")
    subprocess.run(["D:/Pythonproject/.venv/Scripts/python.exe", "train_lstm_model.py"])
    print("✅ Training complete. Sleeping...\n")
    time.sleep(TRAIN_INTERVAL)
