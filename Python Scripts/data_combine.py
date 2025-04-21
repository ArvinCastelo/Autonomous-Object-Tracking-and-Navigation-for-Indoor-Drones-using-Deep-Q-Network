import os
import csv
from datetime import datetime


DATA_DIR = "./dataset10"
OUTPUT_DIR = "./"

def combine_csv_files():
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    combined_data = []

    csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by episode_id

    for csv_file in csv_files:
        # Extract episode_id from the filename (last 3 digits of the filename)
        episode_id = csv_file.split('_')[-1].split('.')[0]  # Extract number between underscores
        file_path = os.path.join(DATA_DIR, csv_file)

        with open(file_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row

            # Read data and prepend episode_id to each row
            for row in reader:
                combined_data.append([episode_id] + row)

    # Write combined data to a new CSV file
    filename = os.path.join(OUTPUT_DIR, f"dataset_{datetime.now():%Y%m%d_%H%M%S}_combined.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        header.insert(0, "episode_id")
        writer.writerow(header)
        writer.writerows(combined_data)

    print(f"Combined CSV saved to {filename}")

if __name__ == "__main__":
    combine_csv_files()
