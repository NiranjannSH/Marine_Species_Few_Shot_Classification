import os
import shutil
import tarfile
import urllib.request
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
    "01": "01_Dascyllus_reticulatus",
    "02": "02_Plectroglyphidodon_dickii",
    "03": "03_Chromis_chrysura",
    "04": "04_Amphiprion_clarkii",
    "05": "05_Chaetodon_lunulatus",
    "06": "06_Chaetodon_trifascialis",
    "07": "07_Myripristis_kuntee",
    "08": "08_Acanthurus_nigrofuscus",
    "09": "09_Hemigymnus_fasciatus",
    "10": "10_Neoniphon_sammara",
    "11": "11_Abudefduf_vaigiensis",
    "12": "12_Canthigaster_valentini",
    "13": "13_Pomacentrus_moluccensis",
    "14": "14_Zebrasoma_scopas",
    "15": "15_Hemigymnus_melapterus",
    "16": "16_Lutjanus_fulvus",
    "17": "17_Scolopsis_bilineata",
    "18": "18_Scaridae",
    "19": "19_Pempheris_vanicolensis",
    "20": "20_Zanclus_cornutus",
    "21": "21_Neoglyphidodon_nigroris",
    "22": "22_Balistapus_undulatus",
    "23": "23_Siganus_fuscescens",
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
    """Downloads the complete Fish4Knowledge dataset archive from Edinburgh's official server."""
    dest_dir = os.path.join(RAW_DIR, "fish4knowledge")
    os.makedirs(dest_dir, exist_ok=True)

    archive_url = "https://homepages.inf.ed.ac.uk/rbf/fish4knowledge/GROUNDTRUTH/RECOG/Archive/fishRecognition_GT.tar"
    dest_file_path = os.path.join(dest_dir, "fishRecognition_GT.tar")

    if os.path.exists(dest_file_path):
        print(f"[Skip Download] Archive already exists at: {dest_file_path}")
        return

    print("[Downloading] Fish4Knowledge complete archive (fishRecognition_GT.tar)...")
    try:
        urllib.request.urlretrieve(archive_url, dest_file_path)
        print("[Download Complete] Saved fishRecognition_GT.tar")
    except Exception as e:
        print(f"[Error Downloading Fish4Knowledge Archive]: {e}")


def prepare_croatian():
    """Moves Croatian Fish Dataset from raw to processed using metadata.xml with SpeciesName_1, SpeciesName_2 format."""
    if DOWNLOAD_DATA:
        download_croatian_raw()

    croatian_raw = os.path.join(RAW_DIR, "croatian")
    xml_path = os.path.join(croatian_raw, "metadata.xml")

    # Search subdirectories if metadata.xml is nested inside Kaggle download structure
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

        # 1-indexed, unpadded format (e.g., 0 -> SpeciesName_1)
        formatted_id = f"{int(folder_num) + 1}"
        target_folder_name = f"{species_name}_{formatted_id}"

        src_folder = os.path.join(croatian_raw, folder_num)
        dest_folder = os.path.join(PROCESSED_DIR, "croatian", target_folder_name)

        if os.path.exists(src_folder):
            os.makedirs(dest_folder, exist_ok=True)
            for fname in os.listdir(src_folder):
                src_file = os.path.join(src_folder, fname)
                if os.path.isfile(src_file):
                    shutil.move(src_file, os.path.join(dest_folder, fname))
            print(f"[Croatian] Moved folder {folder_num} -> {target_folder_name}")

    # Remove raw croatian folder after moving contents
    raw_base_dir = os.path.join(RAW_DIR, "croatian")
    if os.path.exists(raw_base_dir):
        shutil.rmtree(raw_base_dir, ignore_errors=True)


def prepare_fish4knowledge():
    """Extracts fishRecognition_GT.tar archive and dynamically sorts folders into data/processed/fish4knowledge."""
    if DOWNLOAD_DATA:
        download_fish4knowledge_raw()

    f4k_raw = os.path.join(RAW_DIR, "fish4knowledge")
    tar_path = os.path.join(f4k_raw, "fishRecognition_GT.tar")

    if not os.path.exists(f4k_raw):
        print(f"[Skipped] Fish4Knowledge directory not found at: {f4k_raw}")
        return

    # 1. Extract complete archive if present
    if os.path.exists(tar_path):
        print("[Fish4Knowledge] Extracting full dataset archive...")
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=f4k_raw)
        os.remove(tar_path)

    # 2. Walk raw directory and map species folders/archives to processed target directories
    for root_dir, dirs, files in os.walk(f4k_raw, topdown=True):
        for f in list(files):
            if f.endswith(".tar"):
                for pfx, target_name in F4K_SPECIES_MAP.items():
                    if f"fish_{pfx}" in f.lower() or f.startswith(pfx):
                        dest_folder = os.path.join(PROCESSED_DIR, "fish4knowledge", target_name)
                        os.makedirs(dest_folder, exist_ok=True)
                        tar_file_path = os.path.join(root_dir, f)
                        with tarfile.open(tar_file_path, "r:*") as tar:
                            tar.extractall(path=dest_folder)
                        os.remove(tar_file_path)
                        print(f"[Fish4Knowledge] Extracted {f} -> {target_name}")

        for d in list(dirs):
            d_lower = d.lower()
            matched_target = None

            for pfx, target_name in F4K_SPECIES_MAP.items():
                if d_lower == f"fish_{pfx}" or d_lower == pfx or d_lower.startswith(f"{pfx}_"):
                    matched_target = target_name
                    break

            if matched_target:
                src_folder_path = os.path.join(root_dir, d)
                dest_folder_path = os.path.join(PROCESSED_DIR, "fish4knowledge", matched_target)
                os.makedirs(dest_folder_path, exist_ok=True)

                for item in os.listdir(src_folder_path):
                    src_item = os.path.join(src_folder_path, item)
                    dest_item = os.path.join(dest_folder_path, item)
                    shutil.move(src_item, dest_item)

                print(f"[Fish4Knowledge] Moved {d} -> {matched_target}")
                dirs.remove(d)

    # Clean up empty raw directory
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
