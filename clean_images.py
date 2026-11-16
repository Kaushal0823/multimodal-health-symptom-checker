"""
Image Dataset Cleaner (HAM10000 / Skin Disease Photos)
---------------------------------------------------------
Ye script tumhare image dataset (jaise HAM10000) ko clean karta hai
bina kisi upload ke -- seedha tumhare computer/Colab par chalta hai.

Kya-kya karta hai:
1. Corrupt/unreadable images dhundh kar list banata hai (delete nahi karta,
   pehle sirf dikhata hai -- taaki tum check kar sako)
2. Duplicate images dhundta hai (hash-based, exact duplicates)
3. Bahut chhoti (jaise 10x10 pixel) ya galat images flag karta hai
4. Sab images ko ek standard size (jaise 224x224) mein resize karke
   ek naye "cleaned" folder mein save karta hai
5. CSV file ke labels ko images ke saath match karke verify karta hai
   (jo images CSV mein nahi hain, ya jo CSV rows ki images missing hain)
6. Final summary report deta hai

STEP 1 (ek baar): pip install pillow pandas tqdm imagehash --break-system-packages

STEP 2: Neeche CONFIG section mein apne folder/CSV ke naam daalo

STEP 3: python clean_images.py
"""

import os
import shutil
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
import imagehash

# ---------------- CONFIG (tumhare exact folder names ke hisaab se set kar diya hai) ----------------
# Tumhare paas images 2 folders mein split hain, dono ka path yaha do (list mein).
IMAGE_FOLDERS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
METADATA_CSV = "HAM10000_metadata.csv"    # CSV jisme image_id aur label hai
IMAGE_ID_COLUMN = "image_id"              # CSV mein column jisme image ka naam/id hai
LABEL_COLUMN = "dx"                       # CSV mein column jisme disease label hai
IMAGE_EXTENSION = ".jpg"                  # tumhare images ka extension (.jpg / .png)

OUTPUT_FOLDER = "cleaned_images"          # resized, clean images yaha jayengi
TARGET_SIZE = (224, 224)                  # standard size CNN models ke liye (ResNet etc.)
MIN_VALID_SIZE = (20, 20)                 # isse chhoti image = corrupt/useless maano


def find_corrupt_and_tiny_images(folders):
    """Har image kholke check karta hai ki wo valid hai ya nahi. 'folders' ek list hai (multiple folders support)."""
    bad_files = []
    all_files = []
    for folder in folders:
        found = list(Path(folder).glob(f"*{IMAGE_EXTENSION}"))
        print(f"'{folder}' mein mile: {len(found)} images")
        all_files.extend(found)
    print(f"Total images found (sab folders milakar): {len(all_files)}")

    for path in tqdm(all_files, desc="Checking image validity"):
        try:
            with Image.open(path) as img:
                img.verify()  # corrupt check
            with Image.open(path) as img:  # verify() ke baad dubara kholna padta hai
                w, h = img.size
                if w < MIN_VALID_SIZE[0] or h < MIN_VALID_SIZE[1]:
                    bad_files.append((str(path), "too small"))
        except Exception as e:
            bad_files.append((str(path), f"corrupt: {e}"))

    return bad_files, all_files


def find_duplicate_images(all_files):
    """Perceptual hashing se exact/near-duplicate images dhundta hai."""
    hashes = {}
    duplicates = []

    for path in tqdm(all_files, desc="Checking duplicates"):
        try:
            with Image.open(path) as img:
                h = str(imagehash.average_hash(img))
            if h in hashes:
                duplicates.append((str(path), "duplicate of: " + hashes[h]))
            else:
                hashes[h] = str(path)
        except Exception:
            continue  # already flagged as corrupt

    return duplicates


def verify_metadata_match(all_files, csv_path):
    """CSV ke labels aur actual images ka match check karta hai."""
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' nahi mili -- ye step skip kar raha hoon.")
        return

    df = pd.read_csv(csv_path)
    csv_ids = set(df[IMAGE_ID_COLUMN].astype(str))
    file_ids = set(Path(f).stem for f in all_files)

    missing_images = csv_ids - file_ids     # CSV mein hai, image nahi
    extra_images = file_ids - csv_ids        # image hai, CSV mein nahi

    print(f"\nCSV rows: {len(df)}")
    print(f"Images with missing files (in CSV but no image): {len(missing_images)}")
    print(f"Extra images (image exists but not in CSV): {len(extra_images)}")

    if LABEL_COLUMN in df.columns:
        print(f"\nLabel distribution:\n{df[LABEL_COLUMN].value_counts()}")

    return missing_images, extra_images


def clean_and_resize(all_files, bad_files, duplicate_files, output_folder):
    """Corrupt/duplicate images chhod kar baaki sab ko resize karke save karta hai."""
    bad_paths = set(b[0] for b in bad_files)
    dup_paths = set(d[0] for d in duplicate_files)
    skip_paths = bad_paths | dup_paths

    os.makedirs(output_folder, exist_ok=True)
    saved = 0

    for path in tqdm(all_files, desc="Resizing clean images"):
        if str(path) in skip_paths:
            continue
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize(TARGET_SIZE)
                out_path = Path(output_folder) / Path(path).name
                img.save(out_path, quality=95)
                saved += 1
        except Exception:
            continue

    print(f"\nSaved {saved} cleaned + resized images to '{output_folder}'")


def main():
    print("=== STEP 1: Corrupt / tiny image check ===")
    bad_files, all_files = find_corrupt_and_tiny_images(IMAGE_FOLDERS)
    print(f"Found {len(bad_files)} bad images.")
    if bad_files:
        pd.DataFrame(bad_files, columns=["file", "reason"]).to_csv("bad_images_report.csv", index=False)
        print("Details saved to bad_images_report.csv")

    print("\n=== STEP 2: Duplicate check ===")
    duplicate_files = find_duplicate_images(all_files)
    print(f"Found {len(duplicate_files)} duplicate images.")
    if duplicate_files:
        pd.DataFrame(duplicate_files, columns=["file", "duplicate_info"]).to_csv("duplicate_images_report.csv", index=False)
        print("Details saved to duplicate_images_report.csv")

    print("\n=== STEP 3: Metadata (CSV) match check ===")
    verify_metadata_match(all_files, METADATA_CSV)

    print("\n=== STEP 4: Resize + save clean images ===")
    clean_and_resize(all_files, bad_files, duplicate_files, OUTPUT_FOLDER)

    print("\n=== DONE ===")
    print("Ab tum 'cleaned_images' folder ko apne model training mein use kar sakte ho.")


if __name__ == "__main__":
    main()
