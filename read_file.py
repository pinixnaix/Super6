import os
import pandas as pd
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the CSV file
file_path = 'all_matches.csv'
logging.info('Loading the CSV file from %s', file_path)
try:
    data = pd.read_csv(file_path)
    logging.info('CSV file loaded successfully')
except FileNotFoundError:
    logging.error('CSV file not found: %s', file_path)
    exit(1)
except pd.errors.EmptyDataError:
    logging.error('CSV file is empty: %s', file_path)
    exit(1)
except Exception as e:
    logging.error('Error loading CSV file: %s', str(e))
    exit(1)

# Convert the 'date' column to datetime
logging.info('Converting the "date" column to datetime format')
try:
    data['date'] = pd.to_datetime(data['date'], format="mixed")
    logging.info('"date" column converted successfully')
except Exception as e:
    logging.error('Error converting "date" column: %s', str(e))
    exit(1)

# Initialize the InfluxDB client
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "PinaCode")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "Euros2024")

logging.info('Initializing InfluxDB client')
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)
logging.info('InfluxDB client initialized successfully')

# Prepare and write the data points
logging.info('Preparing and writing data points to InfluxDB')
for index, row in data.iterrows():
    try:
        point = Point("match_results") \
            .time(row['date']) \
            .tag("home_team", row['home_team']) \
            .tag("away_team", row['away_team']) \
            .tag("tournament", row['tournament']) \
            .tag("country", row['country']) \
            .tag("neutral", str(row['neutral'])) \
            .field("home_score", row['home_score']) \
            .field("away_score", row['away_score'])

        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        logging.info('Data point %d written successfully', index)
    except Exception as e:
        logging.error('Error writing data point %d: %s', index, str(e))

# Close the client
logging.info('Closing the InfluxDB client')
client.close()
logging.info('InfluxDB client closed successfully')
