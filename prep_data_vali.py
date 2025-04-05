import os
import shutil
import pandas as pd

# Indices you want to extract
indices = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900]

image_src_dir = "./prompt-generation/data/diffusiondb_images"
csv_path = "./prompt-generation/data/diffusiondb_sample.csv"
image_dest_dir = "./data/images"
prompt_dest_csv = "./data/prompts.csv"

os.makedirs(image_dest_dir, exist_ok=True)

# Load the prompt CSV
df = pd.read_csv(csv_path)

# Prepare new DataFrame to store selected prompts
selected_rows = []

for idx in indices:
    # Get the row at the specified index
    row = df.iloc[idx]

    # Get image filename and prompt
    filename = row["file_name"]
    prompt = row["prompt"]

    # Copy image to destination folder
    src_img = os.path.join(image_src_dir, filename)
    dst_img = os.path.join(image_dest_dir, filename)
    shutil.copyfile(src_img, dst_img)

    # Append row to selected prompts
    selected_rows.append({"imgId": os.path.splitext(filename)[0], "prompt": prompt})

# Save selected prompts to CSV in expected format
selected_df = pd.DataFrame(selected_rows)
selected_df.to_csv(prompt_dest_csv, index=False)

print("✅ Images and prompts extracted successfully.")
