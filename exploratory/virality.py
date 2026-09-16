import glob
import pandas as pd

# Find all the CSV datasets you already extracted
CSV_DIR = "/Users/nehal/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/Misinfo-research/Data/CSVs"
csv_files = glob.glob(f"{CSV_DIR}/*_data.csv")
max_m = 0
best_file = ""

for file in csv_files:
    try:
        df = pd.read_csv(file)
        # Find the peak number of Misinformed people in this dataset
        current_max = df['M_cumulative'].max()
        if current_max > max_m:
            max_m = current_max
            best_file = file
    except:
        pass

print(f"🔥 THE MOST VIRAL RUMOR IS: {best_file}")
print(f"Total people infected: {max_m}")