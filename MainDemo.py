import asyncio
from playwright.async_api import async_playwright
import base64
import json
import sqlite3
import os

print("Using DB file at:", os.path.abspath("market_data.db"))

def init_db():
    conn = sqlite3.connect("market_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp REAL,
            price REAL
        )
    """)
    conn.commit()
    return conn

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # Connect to the DevTools protocol
        client = await context.new_cdp_session(page)

        # Enable network tracking
        await client.send("Network.enable")

        # Listen for all WebSocket frame payloads
        client.on("Network.webSocketFrameReceived", lambda event: handle_ws_frame(event))

        # Go to the site
        await page.goto("https://pocketoption.com/en/cabinet/try-demo")
        print("✅ Page loaded. Listening for all WebSocket frames (text + binary)...")

        # Close modal if it appears
        try:
            await page.evaluate('''() => {
            const elements = document.getElementsByClassName('tutorial-v1__close-icon');
            if (elements.length > 0) {
                elements[0].click();
            }
            }''')
            print("✅ Closed modal using JS evaluation.")
        except Exception as e:
            print("❌ No modal to close or timeout occurred:", e)

        # Set the ticket in the local storage in the browser.
        await page.evaluate('''() => {
            localStorage.setItem("po-td-asset", "AUDCAD_otc");
        }''')

        # refresh the page to apply the local storage change
        await page.reload()

        # Close modal if it appears
        try:
            await page.evaluate('''() => {
            const elements = document.getElementsByClassName('tutorial-v1__close-icon');
            if (elements.length > 0) {
                elements[0].click();
            }
            }''')
            print("✅ Closed modal using JS evaluation.")
        except Exception as e:
            print("❌ No modal to close or timeout occurred:", e)

        # Start monitoring signals
        asyncio.create_task(monitor_signals(page))

        while True:
            await asyncio.sleep(1)

async def monitor_signals(page):
    """Monitor the signals.txt file and execute trades based on signals."""
    processed_signals = set()  # Keep track of processed signals to avoid duplicates

    while True:
        try:
            with open("signals.txt", "r") as file:
                lines = file.readlines()

            for line in lines:
                signal, timestamp, price = line.strip().split(",")
                if timestamp not in processed_signals:
                    processed_signals.add(timestamp)

                    if signal == "BUY":
                        print(f"Executing BUY at {price}")
                        await page.click('a.btn.btn-call')  # Selector for the BUY button
                    elif signal == "SELL":
                        print(f"Executing SELL at {price}")
                        await page.click('a.btn.btn-put')  # Selector for the SELL button

        except FileNotFoundError:
            print("signals.txt not found. Waiting for signals...")
        except Exception as e:
            print(f"Error processing signals: {e}")

        await asyncio.sleep(1)  # Check for new signals every second

def handle_ws_frame(event):
    payload = event["response"]["payloadData"]

    try:
        # Try decoding as Base64 (most frames are base64 encoded binary)
        raw_bytes = base64.b64decode(payload + '===')  # add padding if needed
        text = raw_bytes.decode("utf-8")

        # Try parsing JSON structure
        parsed = json.loads(text)
        if isinstance(parsed, list) and isinstance(parsed[0], list):
            for tick in parsed:
                if len(tick) == 3:
                    symbol, timestamp, price = tick
                    print(f"Tick: {symbol} @ {timestamp} = {price}")
                    cursor = db.cursor()
                    cursor.execute(
                        "INSERT INTO ticks (symbol, timestamp, price) VALUES (?, ?, ?)",
                        (symbol, timestamp, price)
                    )
                    db.commit()
        else:
            # Optional: log if it's a message but not tick data
            pass

    except (UnicodeDecodeError, json.JSONDecodeError, base64.binascii.Error):
        # Known bad frames: ignore silently
        pass
    except Exception as e:
        # Unknown error — log it
        print("Unexpected error decoding frame:", e)


db = init_db()
asyncio.run(main())
