import cv2
import random
import shutil
import numpy as np
import albumentations as A
from pathlib import Path
import os
from config import load_config

def split_training_validation(src_dir, dest_train_dir, dest_val_dir):
    """
    Helper function that split the dataset into training and validation dataset based on the configuration split ratio
    """
    config = load_config()
    dataset_config = config["dataset_pipeline"]["ratios"]
    training_ratio = dataset_config["training_ratio"]
    valid_extensions = config["general_configuration"]["valid_extensions"]

    src_path = Path(src_dir)
    dest_train_path = Path(dest_train_dir)
    dest_val_path = Path(dest_val_dir)
    
    directories = [
        d for d in src_path.iterdir() if d.is_dir()
    ]

    for class_dir in directories:
        files = [
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        random.shuffle(files)

        idx_training = int(len(files)*training_ratio)
        training_files = files[:idx_training]
        validation_files = files[idx_training:]

        dest_training_dir = dest_train_path / class_dir.name
        dest_validation_dir = dest_val_path / class_dir.name

        dest_training_dir.mkdir(parents=True, exist_ok=True)
        dest_validation_dir.mkdir(parents=True, exist_ok=True)

        for f in training_files:
            shutil.copy2(f, dest_training_dir/f.name)

        for f in validation_files:
            shutil.copy2(f, dest_validation_dir/f.name)
        

def copy_pool(pool, dest_dir):
    """
    Helper function to copy a pool of image in the destination directory.
    Guarantee the balancing of 'cimossa' class with horizzontal flip.
    """
    for i, (img_path, category) in enumerate(pool):
        if category == "cimossa":
            if i % 2 == 0:
                shutil.copy2(img_path, dest_dir / f"{category}_L_{img_path.name}")
            else:
                img = cv2.imread(str(img_path))
                if img is not None:
                    flipped = cv2.flip(img, 1)
                    cv2.imwrite(str(dest_dir / f"{category}_R_{img_path.name}"), flipped)
        else:
            shutil.copy2(img_path, dest_dir / f"{category}_{img_path.name}")

def apply_dynamic_augmentation(image_path, config):
    """
    Applies a universal augmentation pipeline to fortify the Transfer Learning backbone.
    1. Baseline Geometric: Probabilistic flips and 180-degree rotations.
    2. Domain Randomization (Stress Test): Applied ONLY to a maximum percentage of the images.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    params = config["dataset_pipeline"]["augmentation_params"]

    # ==========================================
    # GEOMETRIC AUGMENTATION
    # Albumentations automatically handles the application probabilities (e.g., 50%)
    # ==========================================
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base_transform = A.Compose([
        A.HorizontalFlip(p=params["prob_h_flip"]),
        A.VerticalFlip(p=params["prob_v_flip"]),
        A.Rotate(
            limit=[180, 180],
            border_mode=cv2.BORDER_REFLECT_101,
            p=params["prob_rot_180"]
        )
    ])
    augmented_base = base_transform(image=image_rgb)["image"]
    image_aug = cv2.cvtColor(augmented_base, cv2.COLOR_RGB2BGR)

    # ==========================================
    # DOMAIN RANDOMIZATION
    # Applied only to a subset of the images based on prob_stress
    # ==========================================
    prob_stress = params.get("prob_stress", 0.5)

    if random.random() < prob_stress:
        height, width = image_aug.shape[:2]
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

        num_waves = random.uniform(*params["textile_waves_range"])
        phase = random.uniform(0, np.pi)
        force = random.uniform(*params["textile_force_range"])

        if random.choice([True, False]):
            # HORIZONTAL STRETCH + ILLUMINATION
            frequency = x_grid / width * np.pi * num_waves + phase
            x_deformed = x_grid + force * np.sin(frequency)

            direction = random.choice([-1.0, 1.0])
            intensity = random.uniform(*params["textile_intensity_range"])
            inclination = np.cos(frequency) * direction
            map_light = 1.0 + (intensity * inclination)

            image_distorted = cv2.remap(
                image_aug, x_deformed.astype(np.float32), y_grid.astype(np.float32),
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            map_light_distortion = cv2.remap(
                map_light.astype(np.float32), x_deformed.astype(np.float32), y_grid.astype(np.float32),
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )

            light_distorted = image_distorted.astype(np.float32) * map_light_distortion[..., np.newaxis]
            return np.clip(light_distorted, 0, 255).astype(np.uint8)

        else:
            # VERTICAL STRETCH
            frequency = y_grid / height * np.pi * num_waves + phase
            y_deformed = y_grid + force * np.sin(frequency)

            image_distorted = cv2.remap(
                image_aug, x_grid.astype(np.float32), y_deformed.astype(np.float32),
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            return image_distorted

    return image_aug

def extract_images_by_category(source_dir, category_list, valid_extensions):
    """
    Scans the source directory and returns a shuffled list of tuples: (image_path, category_name)
    Limits the 'cimossa' category to a maximum of 30% of its original images.
    """
    extracted = []
    source_path = Path(source_dir)
    config = load_config()
    params = config["dataset_pipeline"]["ratios"]

    for category in category_list:
        cat_dir = source_path / category
        if cat_dir.exists() and cat_dir.is_dir():
            images = [f for f in cat_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_extensions]

            # Undersampling for "cimossa"
            if category == "cimossa":
                limit = int(len(images) * params["cimossa_undersampling_ratio"])
                images = random.sample(images, limit)
            extracted.extend([(img, category) for img in images])

    random.shuffle(extracted)
    return extracted

def build_mutually_exclusive_datasets():
    """
    Orchestrates the entire dataset split with a strict ZERO WASTE policy for defects.
    Guarantees no data leakage between Transfer Learning (Phase 1) and Patchcore (Phase 2).
    """
    config = load_config()
    conf = config["dataset_pipeline"]
    valid_extensions = tuple(config["general_configuration"]["valid_extensions"])

    # Paths initialization
    unsplitted_dataset = conf["paths"]["source_unsplitted_dataset"]
    flag_split_source = conf["split_source"]
    src_training = conf["paths"]["source_training"]
    src_validation = conf["paths"]["source_validation"]
    dest_tl = Path(conf["paths"]["dest_transfer_learning"])
    dest_ad = Path(conf["paths"]["dest_patchcore"])

    good_cats = conf["classes"]["good_categories"]
    defect_cats = conf["classes"]["defect_categories"]
    tl_ratio = conf["ratios"]["tl_allocation_ratio"]

    # Destination directories
    tl_train_good_dir = dest_tl / "train" / "good"
    tl_train_reject_dir = dest_tl / "train" / "reject"
    tl_val_good_dir = dest_tl / "val" / "good"
    tl_val_reject_dir = dest_tl / "val" / "reject"

    ad_train_good_dir = dest_ad / "train"
    ad_test_good_dir = dest_ad / "test" / "good"
    ad_test_reject_dir = dest_ad / "test" / "reject"

    for folder in [tl_train_good_dir, tl_train_reject_dir, tl_val_good_dir, tl_val_reject_dir,
                   ad_train_good_dir, ad_test_good_dir, ad_test_reject_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    if flag_split_source:
        print("--- SPLIT SOURCE IN TRAINING AND VALIDATION DIRECTORY ---")
        split_training_validation(unsplitted_dataset, src_training, src_validation)
        print("Source dataset splitted into traingin and validation")


    print("--- EXTRACTING AND SLICING SOURCE DATA (ZERO WASTE POLICY) ---")

    # PROCESS GOOD IMAGES
    train_good_pool = extract_images_by_category(src_training, good_cats, valid_extensions)
    val_good_pool = extract_images_by_category(src_validation, good_cats, valid_extensions)

    num_tl_train_good = int(len(train_good_pool) * tl_ratio)
    tl_train_good = train_good_pool[:num_tl_train_good]
    ad_train_good = train_good_pool[num_tl_train_good:]

    num_tl_val_good = int(len(val_good_pool) * tl_ratio)
    tl_val_good = val_good_pool[:num_tl_val_good]
    ad_test_good = val_good_pool[num_tl_val_good:]

    # PROCESS DEFECT IMAGES
    train_defect_pool = extract_images_by_category(src_training, defect_cats, valid_extensions)
    val_defect_pool = extract_images_by_category(src_validation, defect_cats, valid_extensions)

    num_tl_train_defect = int(len(train_defect_pool) * tl_ratio)
    tl_train_defect = train_defect_pool[:num_tl_train_defect]

    # Pool ALL remaining unused defects from everywhere
    unused_train_defects = train_defect_pool[num_tl_train_defect:]
    all_remaining_defects = unused_train_defects + val_defect_pool

    # ==========================================
    # ROUTING DEFECTS: Balance TL Val first, otherwise split 50/50
    # ==========================================
    target_tl_val = len(tl_val_good)
    total_remaining = len(all_remaining_defects)

    if total_remaining >= target_tl_val:
        # We have enough to perfectly balance TL Val. The rest goes to AD Test.
        tl_val_defect = all_remaining_defects[:target_tl_val]
        ad_test_defect = all_remaining_defects[target_tl_val:]
    else:
        # Not enough to balance TL Val. Split the remaining defects equally 50/50.
        half_point = total_remaining // 2
        tl_val_defect = all_remaining_defects[:half_point]
        ad_test_defect = all_remaining_defects[half_point:]
        print(f"WARNING: Not enough defects to balance TL Validation. "
              f"Splitting remaining {total_remaining} defects equally: "
              f"{len(tl_val_defect)} to TL Val, {len(ad_test_defect)} to AD Test.")

    # Dispatch files to their final destinations
    copy_pool(tl_train_good, tl_train_good_dir)
    copy_pool(tl_train_defect, tl_train_reject_dir)
    copy_pool(tl_val_good, tl_val_good_dir)
    copy_pool(tl_val_defect, tl_val_reject_dir)

    ad_good_limit = conf["ratios"]["good_limit_patchcore"]
    if len(ad_train_good) > ad_good_limit:
        ad_train_good=random.sample(ad_train_good, ad_good_limit)
        copy_pool(ad_train_good, ad_train_good_dir)
    else:
        copy_pool(ad_train_good, ad_train_good_dir)
    copy_pool(ad_test_good, ad_test_good_dir)
    copy_pool(ad_test_defect, ad_test_reject_dir)
    print("Source data extracted and sliced")

    print("\n--- GENERATING INVARIANCE FOR NORMAL IMAGES (STRESS TEST) ---")
    stress_multiplier = conf["ratios"].get("good_stress_multiplier", 1)
    if stress_multiplier > 0:
        print(f"Generating {len(tl_train_good) * stress_multiplier} distorted variants to fortify the backbone...")
        for img_path, category in tl_train_good:
            for i in range(stress_multiplier):
                augmented_img = apply_dynamic_augmentation(img_path, config)
                if augmented_img is not None:
                    new_filename = f"aug_stress_{i}_{category}_{img_path.name}"
                    cv2.imwrite(str(tl_train_good_dir / new_filename), augmented_img)

    print("\n--- TRANSFER LEARNING CLASS BALANCING ---")
    target_good_count = len(list(tl_train_good_dir.glob("*")))
    current_reject_count = len(list(tl_train_reject_dir.glob("*")))

    print(f"TL Train Set -> Good (including stress variants): {target_good_count} | Defects: {current_reject_count}")

    if 0 < current_reject_count < target_good_count:
        missing_images = target_good_count - current_reject_count
        print(f"Imbalance detected. Synthesizing {missing_images} defect variants...")

        generated_count = 0
        while generated_count < missing_images:
            base_img_path, defect_category = random.choice(tl_train_defect)
            augmented_img = apply_dynamic_augmentation(base_img_path, config)

            if augmented_img is not None:
                new_filename = f"aug_{generated_count}_{defect_category}_{base_img_path.name}"
                cv2.imwrite(str(tl_train_reject_dir / new_filename), augmented_img)
                generated_count += 1

        print("Defect balancing completed successfully!")

    print("\n--- DATASETS READY FOR TRAINING ---")
    print(f"AD Test Set -> Good: {len(ad_test_good)} | Defects (Zero Waste): {len(ad_test_defect)}")

if __name__ == "__main__":
    build_mutually_exclusive_datasets()
