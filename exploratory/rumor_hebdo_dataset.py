import os
import json
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
# Point this to the ROOT dataset folder that contains ALL the events
PHEME_ROOT_DIR = "/Users/nehal/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/Misinfo-research/Data/Mis-pheme-rnr-dataset"
TOP_N_PER_EVENT = 3 # How many top rumors to pull from EACH event
OUTPUT_DIR = "/Users/nehal/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/Misinfo-research/Data/CSVs"  # where the per-rumor CSVs are written
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_twitter_date(date_str):
    return datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')

# 1. Scan through every Event folder (charliehebdo, ferguson, etc.)
if os.path.exists(PHEME_ROOT_DIR):
    for event_name in os.listdir(PHEME_ROOT_DIR):
        event_path = os.path.join(PHEME_ROOT_DIR, event_name)
        rumours_base_dir = os.path.join(event_path, 'rumours')
        
        # Skip hidden files or folders that aren't actual event directories
        if not os.path.isdir(rumours_base_dir):
            continue
            
        print(f"\n========================================")
        print(f"🚀 SCANNING EVENT: {event_name.upper()}")
        print(f"========================================")

        folder_sizes = {}
        for folder_name in os.listdir(rumours_base_dir):
            reactions_dir = os.path.join(rumours_base_dir, folder_name, 'reactions')
            if os.path.exists(reactions_dir) and os.path.isdir(reactions_dir):
                num_files = len([f for f in os.listdir(reactions_dir) if os.path.isfile(os.path.join(reactions_dir, f))])
                folder_sizes[folder_name] = num_files

        # Sort and get the Top N for this specific event
        sorted_folders = sorted(folder_sizes.items(), key=lambda x: x[1], reverse=True)[:TOP_N_PER_EVENT]

        # 2. Process the Top Rumors for this Event
        for rumor_id, size in sorted_folders:
            print(f"  -> Processing Rumor ID: {rumor_id} ({size} reactions)...")
            rumor_folder_path = os.path.join(rumours_base_dir, rumor_id)
            
            sdqc_labels = {}
            annotation_path = os.path.join(rumor_folder_path, 'annotation.json')
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    try:
                        anno_data = json.load(f)
                        if 'responses' in anno_data:
                            sdqc_labels = anno_data['responses']
                    except:
                        pass

            tweets_data = []

            def process_subfolder(folder_path):
                if os.path.exists(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith('.json'):
                            with open(os.path.join(folder_path, file), 'r') as f:
                                data = json.load(f)
                                tweet_id = data.get('id_str', '')
                                text = data.get('text', '').lower()
                                created_at = data.get('created_at', '')
                                
                                if not created_at: continue
                                timestamp = parse_twitter_date(created_at)
                                
                                label = sdqc_labels.get(tweet_id, None)
                                if not label:
                                    deny_keywords = ['fake', 'false', 'hoax', 'lie', 'bullshit', 'not true', 'debunk']
                                    if any(kw in text for kw in deny_keywords):
                                        label = 'deny'
                                    else:
                                        label = 'support'
                                        
                                if label == 'deny': compartment = 'C'
                                elif label == 'support': compartment = 'M'
                                else: compartment = 'S'
                                    
                                tweets_data.append({'timestamp': timestamp, 'compartment': compartment})

            process_subfolder(os.path.join(rumor_folder_path, 'source-tweets'))
            process_subfolder(os.path.join(rumor_folder_path, 'reactions'))

            if len(tweets_data) > 0:
                df = pd.DataFrame(tweets_data)
                df.sort_values('timestamp', inplace=True)
                
                t0 = df['timestamp'].iloc[0]
                df['time_hour'] = ((df['timestamp'] - t0).dt.total_seconds() / 3600.0).astype(int)
                
                hourly_counts = df.groupby(['time_hour', 'compartment']).size().unstack(fill_value=0)
                
                if 'M' not in hourly_counts.columns: hourly_counts['M'] = 0
                if 'C' not in hourly_counts.columns: hourly_counts['C'] = 0
                
                hourly_counts['M_cumulative'] = hourly_counts['M'].cumsum()
                hourly_counts['C_cumulative'] = hourly_counts['C'].cumsum()
                hourly_counts = hourly_counts.reset_index()
                
                max_hour = hourly_counts['time_hour'].max()
                all_hours = pd.DataFrame({'time_hour': range(0, max_hour + 1)})
                merged = pd.merge(all_hours, hourly_counts, on='time_hour', how='left')
                
                merged['M_cumulative'] = merged['M_cumulative'].ffill().fillna(0).astype(int)
                merged['C_cumulative'] = merged['C_cumulative'].ffill().fillna(0).astype(int)
                
                # Name the CSV with both the Event and the ID so you know what it is!
                output_file = os.path.join(OUTPUT_DIR, f'{event_name}_rumor_{rumor_id}_data.csv')
                merged[['time_hour', 'M_cumulative', 'C_cumulative']].to_csv(output_file, index=False)
else:
    print("Could not find the root dataset folder. Double check your PHEME_ROOT_DIR path!")

print("\n✅ ALL EVENTS PROCESSED SUCCESSFULLY!")