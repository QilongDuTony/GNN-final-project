import os
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

print("Loading DiffusionDB (full image version)...")
dataset = load_dataset("poloclub/diffusiondb", "large_random_1k")

samples = dataset["train"]
save_dir = "./data/diffusiondb_images"
os.makedirs(save_dir, exist_ok=True)
csv_path = "./data/diffusiondb_sample.csv"
records = []

print("Saving images and prompts...")
for i, item in tqdm(enumerate(samples), total=100):
    image = item["image"]
    prompt = item["prompt"]

    file_name = f"image_{i:04d}.png"
    image_path = os.path.join(save_dir, file_name)
    try:
        image.save(image_path)
        records.append({"file_name": file_name, "prompt": prompt})
    except Exception as e:
        print(f"Failed to save image {i}: {e}")
print("Saving CSV...")
pd.DataFrame(records).to_csv(csv_path, index=False)
print(f"Saved CSV to {csv_path}")
print("Done.")
