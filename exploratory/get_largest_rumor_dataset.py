import os

# --- CONFIGURATION ---
# Paste the absolute path to your local 'rumours' folder here.
# Windows example: r"C:\Users\YourName\Downloads\charliehebdo\rumours"
# Mac/Linux example: "/Users/YourName/Downloads/charliehebdo/rumours"
RUMOURS_BASE_DIR = r"/Users/nehal/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/Misinfo-research/Data/Mis-pheme-rnr-dataset/charliehebdo/rumours"

folder_sizes = {}

if os.path.exists(RUMOURS_BASE_DIR):
    print("Scanning for viral rumors on your local machine...")
    
    for folder_name in os.listdir(RUMOURS_BASE_DIR):
        reactions_dir = os.path.join(RUMOURS_BASE_DIR, folder_name, 'reactions')
        
        # Check if the reactions directory actually exists
        if os.path.exists(reactions_dir) and os.path.isdir(reactions_dir):
            # Count how many files are in the reactions folder (ignoring hidden OS folders)
            num_files = len([f for f in os.listdir(reactions_dir) if os.path.isfile(os.path.join(reactions_dir, f))])
            folder_sizes[folder_name] = num_files

    # Sort the folders by the number of reactions (largest first)
    sorted_folders = sorted(folder_sizes.items(), key=lambda x: x[1], reverse=True)

    print("\n🔥 Top 5 Biggest Rumors in Your Dataset 🔥")
    for folder, size in sorted_folders[:5]:
        print(f"Folder ID: {folder} | Total Reactions: {size}")
else:
    print(f"Uh oh, couldn't find the directory: {RUMOURS_BASE_DIR}\nDouble-check your folder path!")