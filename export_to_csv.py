import sqlite3
import pandas as pd

# Connect to your SQLite database
conn = sqlite3.connect('market_data.db')

# Define your SQL query to select data
query = """
    SELECT timestamp, price
    FROM ticks
    WHERE symbol = 'AUDCAD_otc'
    ORDER BY timestamp ASC
"""

# Execute the query and load data into a pandas DataFrame
df = pd.read_sql_query(query, conn)

# Export the DataFrame to a CSV file
df.to_csv('tick_data.csv', index=False)

# Close the database connection
conn.close()

print(" Tick data has been successfully exported to 'tick_data.csv'.")
