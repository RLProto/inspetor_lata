import os
import time
import asyncio
import aiohttp
import logging
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import shutil
from influxdb import InfluxDBClient  # Add this import to the top of your script
import requests.exceptions  # Add this import at the top of your script
from watchdog.observers.polling import PollingObserver as Observer


# Setup InfluxDB connection with a timeout
influx_client = InfluxDBClient(
    host='10.15.160.2', 
    port=8086, 
    database='L511', 
    timeout=2  # Set a 5-second timeout for the connection
)

# Constants
INFERENCE_ENDPOINT = "/inference"
UPLOAD_MODEL_ENDPOINT = "/upload-model"
URL_9999 = os.getenv('URL_9999', 'http://localhost:9999')
WATCH_FOLDER = r"./teste_inspetor"
INFER_RATE_LIMIT = 2  # in seconds
MODEL_PATH = os.getenv('MODEL_PATH',"./model/cinta.zip")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageHandler(FileSystemEventHandler):
    def __init__(self, q, initial_delay=1.0):
        self.q = q
        self.initial_delay = initial_delay  # Delay in seconds before processing

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.jpg', '.jpeg', '.png')):
            logging.info(f"New image detected: {event.src_path}")
            time.sleep(self.initial_delay)  # Wait for a short period before queueing
            self.q.put(event.src_path)

async def upload_model(session, url, file_path):
    with open(file_path, 'rb') as file:
        data = aiohttp.FormData()
        data.add_field('file', file, filename=os.path.basename(file_path))
        try:
            response = await session.post(url, data=data)
            if response.status == 200:
                logging.info("Model uploaded successfully")
            else:
                logging.error(f"Failed to upload model to {url}: {response.status}")
        except Exception as e:
            logging.error(f"Error during model upload to {url}: {str(e)}")

async def send_request(session, img_path):
    retries = 5  # Increase the number of retries
    initial_delay = 1  # Add a small delay before the first attempt
    await asyncio.sleep(initial_delay)
    
    while retries > 0:
        if is_file_ready(img_path):
            try:
                with open(img_path, 'rb') as file:
                    data = aiohttp.FormData()
                    data.add_field('file', file, filename=os.path.basename(img_path))
                    async with session.post(URL_9999 + INFERENCE_ENDPOINT, data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result
                        else:
                            logging.error(f"Failed to get a valid response from {URL_9999}: {response.status}")
            except Exception as e:
                logging.error(f"Error opening file {img_path}: {e}")
        else:
            logging.info(f"File {img_path} not ready, retrying...")
        
        retries -= 1
        await asyncio.sleep(1)  # Increase the sleep time or make it dynamic based on retries
    logging.error(f"File {img_path} not ready after several retries, giving up.")
    return None

def is_file_ready(filename):
    """ Check if a file is ready to be opened. """
    try:
        with open(filename, 'rb') as f:
            return True
    except IOError as e:
        logging.info(f"File not ready: {e}")
        return False
    
async def process_images(q):
    async with aiohttp.ClientSession() as session:
        processed_folder_path = os.path.join(WATCH_FOLDER, "processed")
        os.makedirs(processed_folder_path, exist_ok=True)  # Ensure the processed folder exists

        while True:
            img_path = q.get()
            if img_path:
                logging.info(f"Processing image: {img_path}")
                response = await send_request(session, img_path)
                if response:
                    logging.info(f"Response: {response}")

                    data_points = [{
                        "measurement": "inspetor_lata_amassada",
                        "tags": {
                            # Other tags can still be added here if needed
                        },
                        "fields": {
                            "image_name": response.get('image_name', 'unknown'),  # Now a field
                            "prediction": response.get('prediction', 'unknown'),
                            "accuracy": float(response.get('accuracy', 0.0))
                        }
                    }]

                    # Write data to InfluxDB
                    try:
                        influx_client.write_points(data_points)
                        logging.info("Data written to InfluxDB successfully")
                    except requests.exceptions.Timeout as e:
                        logging.error(f"InfluxDB write timed out: {e}")
                        # Continue processing without interrupting
                    except Exception as e:
                        logging.error(f"Failed to write data to InfluxDB: {e}")

                    # Move processed file
                    try:
                        new_path = os.path.join(processed_folder_path, os.path.basename(img_path))
                        shutil.move(img_path, new_path)
                        logging.info(f"Moved {img_path} to {new_path}")
                    except Exception as e:
                        logging.error(f"Failed to move {img_path} to {new_path}: {e}")

                await asyncio.sleep(INFER_RATE_LIMIT)  # Limit the rate of processing
async def main():
    q = queue.Queue()
    event_handler = ImageHandler(q)
    observer = Observer()  # Using PollingObserver instead of the default Observer
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()

    async with aiohttp.ClientSession() as session:
        # Run model upload once at the start
        await upload_model(session, URL_9999 + UPLOAD_MODEL_ENDPOINT, MODEL_PATH)

        try:
            await process_images(q)
        except KeyboardInterrupt:
            observer.stop()
    observer.join()

if __name__ == "__main__":
    asyncio.run(main())
