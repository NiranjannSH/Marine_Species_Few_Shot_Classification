import os
import shutil
import tarfile
import xml.etree.ElementTree as ET
from datasets import load_dataset, concatenate_datasets

# =====================================================================
# CONFIGURATION FLAG
# Set DOWNLOAD_DATA = True  -> Automatically downloads datasets first
# Set DOWNLOAD_DATA = False -> Processes raw data already in 'data/raw/'
# =====================================================================
DOWNLOAD_DATA = True

# Base directories
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# -------------------------------------------------------------------
# Fish4Knowledge Mapping
# -------------------------------------------------------------------
F4K_SPECIES_MAP = {
    "fish_01": "01_Dascyllus_reticulatus",
    "fish_02": "02_Plectroglyphidodon_dickii",
    "fish_03": "03_Chromis_chrysura",
    "fish_04": "04_Amphiprion_clarkii",
    "fish_05": "05_Chaetodon_lunulatus",
    "fish_06": "06_Chaetodon_trifascialis",
    "fish_07": "07_Myripristis_kuntee",
    "fish_08": "08_Acanthurus_nigrofuscus",
    "fish_09": "09_Hemigymnus_fasciatus",
    "fish_10": "10_Neoniphon_sammara",
    "fish_11": "11_Abudefduf_vaigiensis",
    "fish_12": "12_Canthigaster_valentini",
    "fish_13": "13_Pomacentrus_moluccensis",
    "fish_14": "14_Zebrasoma_scopas",
    "fish_15": "15_Hemigymnus_melapterus",
    "fish_16": "16_Lutjanus_fulvus",
    "fish_17": "17_Scolopsis_bilineata",
    "fish_18": "18_Scaridae",
    "fish_19": "19_Pempheris_vanicolensis",
    "fish_20": "20_Zanclus_cornutus",
    "fish_21": "21_Neoglyphidodon_nigroris",
    "fish_22": "22_Balistapus_undulatus",
    "fish_23": "23_Siganus_fuscescens",
}


def download_croatian_raw():
    """Downloads Croatian Fish dataset from Kaggle via kagglehub into RAW_DIR."""
    dest_dir = os.path.join(RAW_DIR, "croatian")
    if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
        print(f"[Skip Download] Croatian raw data already exists at: {dest_dir}")
        return

    print("[Downloading] Croatian Dataset via kagglehub...")
    try:
        import kagglehub
        download_path = kagglehub.dataset_download("ashfaqsyed/croatian-fish-dataset")
        os.makedirs(dest_dir, exist_ok=True)
        
        for item in os.listdir(download_path):
            src_path = os.path.join(download_path, item)
            dst_path = os.path.join(dest_dir, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)
        print(f"[Download Complete] Croatian raw data saved to {dest_dir}")
    except Exception as e:
        print(f"[Error Downloading Croatian Dataset]: {e}")


def download_fish4knowledge_raw():
    """Downloads Fish4Knowledge dataset from Kaggle mirror via kagglehub into RAW_DIR."""
    dest_dir = os.path.join(RAW_DIR, "fish4knowledge")
    if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
        print(f"[Skip Download] Fish4Knowledge raw data already exists at: {dest_dir}")
        return

    print("[Downloading] Fish4Knowledge Dataset via kagglehub...")
    try:
        import kagglehub
        download_path = kagglehub.dataset_download("nihiremath/fish4knowledge")
        os.makedirs(dest_dir, exist_ok=True)

        for item in os.listdir(download_path):
            src_path = os.path.join(download_path, item)
            dst_path = os.path.join(dest_dir, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)
        print(f"[Download Complete] Fish4Knowledge raw data saved to {dest_dir}")
    except Exception as e:
        print(f"[Error Downloading Fish4Knowledge Dataset]: {e}")


def prepare_croatian():
    """Processes Croatian Fish Dataset using metadata.xml and formats folder names as SpeciesName_1, SpeciesName_2, etc."""
    if DOWNLOAD_DATA:
        download_croatian_raw()

    croatian_raw = os.path.join(RAW_DIR, "croatian")
    xml_path = os.path.join(croatian_raw, "metadata.xml")

    # Search subdirectories if metadata.xml is nested
    if not os.path.exists(xml_path):
        for root_dir, _, files in os.walk(croatian_raw):
            if "metadata.xml" in files:
                xml_path = os.path.join(root_dir, "metadata.xml")
                croatian_raw = root_dir
                break

    if not os.path.exists(xml_path):
        print(f"[Skipped] Croatian Dataset metadata not found at: {xml_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for elem in root.findall("folder"):
        folder_num = elem.attrib["folder_name"]
        species_name = elem.attrib["species_name"].replace(" ", "_")

        formatted_id = f"{int(folder_num) + 1}"
        target_folder_name = f"{species_name}_{formatted_id}"

        src_folder = os.path.join(croatian_raw, folder_num)
        dest_folder = os.path.join(PROCESSED_DIR, "croatian", target_folder_name)

        if os.path.exists(src_folder):
            os.makedirs(dest_folder, exist_ok=True)
            for fname in os.listdir(src_folder):
                src_file = os.path.join(src_folder, fname)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, os.path.join(dest_folder, fname))
            print(f"[Croatian] Processed folder {folder_num} -> {target_folder_name}")


def prepare_fish4knowledge():
    """Dynamically finds and moves Fish4Knowledge folders/files from data/raw to data/processed."""
    if DOWNLOAD_DATA:
        download_fish4knowledge_raw()

    f4k_raw = os.path.join(RAW_DIR, "fish4knowledge")

    if not os.path.exists(f4k_raw):
        print(f"[Skipped] Fish4Knowledge directory not found at: {f4k_raw}")
        return

    # Map numerical prefix ("01", "02", ...) to standard target species name
    prefix_map = {f"{i:02d}": formatted_name for i, formatted_name in enumerate(F4K_SPECIES_MAP.values(), start=1)}

    # Recursively search f4k_raw to handle nested download directories and case variations
    for root_dir, dirs, files in os.walk(f4k_raw, topdown=True):
        for d in list(dirs):
            d_lower = d.lower()
            matched_prefix = None

            # 1. Check for numerical prefix match (e.g., '01_...', '02_...')
            for pfx in prefix_map.keys():
                if d_lower.startswith(f"{pfx}_") or d_lower == pfx:
                    matched_prefix = pfx
                    break

            # 2. Check for legacy 'fish_01' matching
            if not matched_prefix:
                for key_id, target_name in F4K_SPECIES_MAP.items():
                    if d_lower == key_id.lower():
                        matched_prefix = target_name[:2]
                        break

            if matched_prefix:
                target_name = prefix_map[matched_prefix]
                src_folder_path = os.path.join(root_dir, d)
                dest_folder_path = os.path.join(PROCESSED_DIR, "fish4knowledge", target_name)

                os.makedirs(dest_folder_path, exist_ok=True)

                # Move contents into processed directory
                for item in os.listdir(src_folder_path):
                    src_item = os.path.join(src_folder_path, item)
                    dest_item = os.path.join(dest_folder_path, item)
                    shutil.move(src_item, dest_item)

                print(f"[Fish4Knowledge] Moved {d} -> {target_name}")
                dirs.remove(d)

        # Process tar files if present
        for f in list(files):
            if f.endswith(".tar"):
                f_lower = f.lower()
                matched_prefix = None
                for key_id, target_name in F4K_SPECIES_MAP.items():
                    if f_lower.startswith(key_id.lower()):
                        matched_prefix = target_name[:2]
                        break
                if matched_prefix:
                    target_name = prefix_map[matched_prefix]
                    tar_path = os.path.join(root_dir, f)
                    dest_folder_path = os.path.join(PROCESSED_DIR, "fish4knowledge", target_name)
                    os.makedirs(dest_folder_path, exist_ok=True)
                    with tarfile.open(tar_path, "r:*") as tar:
                        tar.extractall(path=dest_folder_path)
                    os.remove(tar_path)
                    print(f"[Fish4Knowledge] Extracted & removed archive {f} -> {target_name}")

    # Remove raw directory after contents are moved
    if os.path.exists(f4k_raw):
        shutil.rmtree(f4k_raw, ignore_errors=True)


def prepare_aqua20():
    """Downloads taufiktrf/AQUA20 via Hugging Face, merges train/test splits,
    excludes the 'diver' class, and saves images directly to data/processed/aqua20."""
    print("[Aqua20] Fetching AQUA20 from Hugging Face (taufiktrf/AQUA20)...")
    try:
        dataset = load_dataset("taufiktrf/AQUA20")
    except Exception as e:
        print(f"[Skipped] Could not load AQUA20 dataset: {e}")
        return

    class_names = dataset["train"].features["label"].names
    merged_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    total_images = len(merged_dataset)

    print(f"[Aqua20] Merged train & test splits ({total_images} total images). Saving class-wise...")

    skipped_count = 0
    saved_count = 0

    for idx, example in enumerate(merged_dataset):
        image = example["image"]
        label_idx = example["label"]
        raw_class_name = class_names[label_idx]

        if raw_class_name.strip().lower() == "diver":
            skipped_count += 1
            continue

        species_name = raw_class_name.replace(" ", "_")
        dest_folder = os.path.join(PROCESSED_DIR, "aqua20", species_name)
        os.makedirs(dest_folder, exist_ok=True)

        img_path = os.path.join(dest_folder, f"aqua20_{idx:05d}.jpg")
        image.convert("RGB").save(img_path, "JPEG")
        saved_count += 1

    print(f"[Aqua20] Completed processing AQUA20 (Saved {saved_count} images, skipped {skipped_count} 'Diver' images).")


if __name__ == "__main__":
    print(f"Starting dataset processing (DOWNLOAD_DATA={DOWNLOAD_DATA})...\n")
    prepare_croatian()
    prepare_fish4knowledge()
    prepare_aqua20()
    print("\nAll datasets successfully processed into 'data/processed/'!")