import os
import glob
import pandas as pd
import re

PREDICTION_FILE = "submission_flow_jepa.csv"   
TEST_IMAGES_DIR = "./dataset/testing_videos"  
OUTPUT_FILE = "jepa_flow2.csv"

def normalize_id(id_str):
    # Split by underscore
    parts = id_str.split('_')
    if len(parts) == 2:
        vid = int(parts[0])
        frame = int(parts[1])
        return f"{vid}_{frame}"

    return id_str

valid_ids = set()

for root, dirs, files in os.walk(TEST_IMAGES_DIR):
    for filename in files:
        if filename.endswith(".jpg"):
            # Folder name
            folder_name = os.path.basename(root)
            vid_match = re.search(r'\d+', folder_name)
            
            # Filename
            frame_match = re.search(r'\d+', filename)
            
            if vid_match and frame_match:
                vid_num = int(vid_match.group())  
                frame_num = int(frame_match.group()) 
                
                # Standard format
                valid_ids.add(f"{vid_num}_{frame_num}")

print(f"Found {len(valid_ids)} video frames")

# Format according to kaggle
df = pd.read_csv(PREDICTION_FILE)


df['Id'] = df['Id'].apply(normalize_id)

#Filter
df_clean = df[df['Id'].isin(valid_ids)].copy()

# Fill Missing 
existing_ids = set(df_clean['Id'].unique())
missing_ids = valid_ids - existing_ids

if missing_ids:
    # Fill missing with 0.0
    missing_data = [{'Id': mid, 'Predicted': 0.0} for mid in missing_ids]
    df_missing = pd.DataFrame(missing_data)
    df_clean = pd.concat([df_clean, df_missing], ignore_index=True)


df_clean['vid'] = df_clean['Id'].apply(lambda x: int(x.split('_')[0]))
df_clean['frame'] = df_clean['Id'].apply(lambda x: int(x.split('_')[1]))
df_clean = df_clean.sort_values(by=['vid', 'frame']).drop(columns=['vid', 'frame'])

df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to {OUTPUT_FILE}")
print(df_clean.head())
