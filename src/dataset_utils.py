import cv2
import random
import shutil
import numpy as np
import albumentations as A
from pathlib import Path
from src.config import load_config

def split_training_validation(src_dir, dest_train_dir, dest_val_dir):
    """
    Splits the dataset into training and validation based on configuration.
    """
    config = load_config()
    dataset_config = config["dataset_pipeline"]["ratios"]
    training_ratio = dataset_config["training_ratio"]
    valid_extensions = config["general_configuration"]["valid_extensions"]

    src_path = Path(src_dir)
    dest_train_path = Path(dest_train_dir)
    dest_val_path = Path(dest_val_dir)
    
    directories = [d for d in src_path.iterdir() if d.is_dir()]

    for class_dir in directories:
        files = [
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        random.shuffle(files)

        idx_training = int(len(files) * training_ratio)
        training_files = files[:idx_training]
        validation_files = files[idx_training:]

        dest_training_dir = dest_train_path / class_dir.name
        dest_validation_dir = dest_val_path / class_dir.name

        dest_training_dir.mkdir(parents=True, exist_ok=True)
        dest_validation_dir.mkdir(parents=True, exist_ok=True)

        for f in training_files:
            shutil.copy2(f, dest_training_dir / f.name)
        for f in validation_files:
            shutil.copy2(f, dest_validation_dir / f.name)

def copy_pool(pool, dest_dir):
    """
    Copies images to destination. Handles 'cimossa' horizontal flip balancing.
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
    Applies geometric and domain randomization (stress test) augmentations.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    params = config["dataset_pipeline"]["augmentation_params"]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    base_transform = A.Compose([
        A.HorizontalFlip(p=params["prob_h_flip"]),
        A.VerticalFlip(p=params["prob_v_flip"]),
        A.Rotate(limit=[180, 180], border_mode=cv2.BORDER_REFLECT_101, p=params["prob_rot_180"])
    ])
    
    augmented_base = base_transform(image=image_rgb)["image"]
    image_aug = cv2.cvtColor(augmented_base, cv2.COLOR_RGB2BGR)

    prob_stress = params.get("prob_stress", 0.5)
    if random.random() < prob_stress:
        height, width = image_aug.shape[:2]
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        num_waves = random.uniform(*params["textile_waves_range"])
        phase = random.uniform(0, np.pi)
        force = random.uniform(*params["textile_force_range"])

        if random.choice([True, False]):
            frequency = x_grid / width * np.pi * num_waves + phase
            x_deformed = x_grid + force * np.sin(frequency)
            direction = random.choice([-1.0, 1.0])
            intensity = random.uniform(*params["textile_intensity_range"])
            inclination = np.cos(frequency) * direction
            map_light = 1.0 + (intensity * inclination)

            image_distorted = cv2.remap(image_aug, x_deformed.astype(np.float32), y_grid.astype(np.float32),
                                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            map_light_distortion = cv2.remap(map_light.astype(np.float32), x_deformed.astype(np.float32), y_grid.astype(np.float32),
                                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            light_distorted = image_distorted.astype(np.float32) * map_light_distortion[..., np.newaxis]
            return np.clip(light_distorted, 0, 255).astype(np.uint8)
        else:
            frequency = y_grid / height * np.pi * num_waves + phase
            y_deformed = y_grid + force * np.sin(frequency)
            return cv2.remap(image_aug, x_grid.astype(np.float32), y_deformed.astype(np.float32),
                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return image_aug

def extract_images_by_category(source_dir, category_list, valid_extensions):
    """
    Returns list of (path, category). Implements undersampling for 'cimossa'.
    """
    extracted = []
    source_path = Path(source_dir)
    config = load_config()
    params = config["dataset_pipeline"]["ratios"]

    for category in category_list:
        cat_dir = source_path / category
        if cat_dir.exists() and cat_dir.is_dir():
            images = [f for f in cat_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_extensions]
            if category == "cimossa":
                limit = int(len(images) * params["cimossa_undersampling_ratio"])
                images = random.sample(images, limit)
            extracted.extend([(img, category) for img in images])

    random.shuffle(extracted)
    return extracted

def build_mutually_exclusive_datasets():
    config = load_config()
    conf = config["dataset_pipeline"]
    tl_config = config["transfer_learning"]
    valid_extensions = tuple(config["general_configuration"]["valid_extensions"])

    # Paths
    dest_tl = Path(conf["paths"]["dest_transfer_learning"])
    dest_ad = Path(conf["paths"]["dest_patchcore"])
    tl_train_good_dir = dest_tl / "train" / "good"
    tl_train_reject_dir = dest_tl / "train" / "reject"
    tl_val_good_dir = dest_tl / "test" / "good"
    tl_val_reject_dir = dest_tl / "test" / "reject"
    ad_train_good_dir = dest_ad / "train"
    ad_test_good_dir = dest_ad / "test" / "good"
    ad_test_reject_dir = dest_ad / "test" / "reject"

    for folder in [tl_train_good_dir, tl_train_reject_dir, tl_val_good_dir, tl_val_reject_dir,
                    ad_train_good_dir, ad_test_good_dir, ad_test_reject_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    if conf["split_source"]:
        split_training_validation(conf["paths"]["source_unsplitted_dataset"], 
                                  conf["paths"]["source_training"], 
                                  conf["paths"]["source_validation"])

    # Processing Good Images
    train_good_pool = extract_images_by_category(conf["paths"]["source_training"], conf["classes"]["good_categories"], valid_extensions)
    val_good_pool = extract_images_by_category(conf["paths"]["source_validation"], conf["classes"]["good_categories"], valid_extensions)

    num_tl_train_good = int(len(train_good_pool) * conf["ratios"]["tl_allocation_ratio"])
    tl_train_good = train_good_pool[:num_tl_train_good]
    ad_train_good = train_good_pool[num_tl_train_good:]

    num_tl_val_good = int(len(val_good_pool) * conf["ratios"]["tl_allocation_ratio"])
    tl_val_good = val_good_pool[:num_tl_val_good]
    ad_test_good = val_good_pool[num_tl_val_good:]

    # Test Only Categories
    if conf["classes"]["test_only_good_categories"]:
        test_only_pool = extract_images_by_category(conf["paths"]["source_training"], conf["classes"]["test_only_good_categories"], valid_extensions)
        test_only_pool += extract_images_by_category(conf["paths"]["source_validation"], conf["classes"]["test_only_good_categories"], valid_extensions)
        ad_test_good.extend(test_only_pool)

    # Processing Defect Images
    train_defect_pool = extract_images_by_category(conf["paths"]["source_training"], conf["classes"]["defect_categories"], valid_extensions)
    val_defect_pool = extract_images_by_category(conf["paths"]["source_validation"], conf["classes"]["defect_categories"], valid_extensions)

    num_tl_train_defect = int(len(train_defect_pool) * conf["ratios"]["tl_allocation_ratio"])
    tl_train_defect = train_defect_pool[:num_tl_train_defect]
    
    all_remaining_defects = train_defect_pool[num_tl_train_defect:] + val_defect_pool
    target_tl_val = len(tl_val_good)

    if len(all_remaining_defects) >= target_tl_val:
        tl_val_defect = all_remaining_defects[:target_tl_val]
        ad_test_defect = all_remaining_defects[target_tl_val:]
    else:
        half = len(all_remaining_defects) // 2
        tl_val_defect = all_remaining_defects[:half]
        ad_test_defect = all_remaining_defects[half:]

    # Dispatch to Folders
    copy_pool(tl_train_good, tl_train_good_dir)
    copy_pool(tl_train_defect, tl_train_reject_dir)
    copy_pool(tl_val_good, tl_val_good_dir)
    copy_pool(tl_val_defect, tl_val_reject_dir)
    
    ad_limit = conf["ratios"]["good_limit_patchcore"]
    ad_train_to_copy = random.sample(ad_train_good, min(len(ad_train_good), ad_limit))
    copy_pool(ad_train_to_copy, ad_train_good_dir)
    copy_pool(ad_test_good, ad_test_good_dir)
    copy_pool(ad_test_defect, ad_test_reject_dir)

    # Stress Test Generation
    stress_mult = conf["ratios"].get("good_stress_multiplier", 1)
    if stress_mult > 0:
        for img_path, cat in tl_train_good:
            for i in range(stress_mult):
                aug = apply_dynamic_augmentation(img_path, config)
                if aug is not None:
                    cv2.imwrite(str(tl_train_good_dir / f"aug_stress_{i}_{cat}_{img_path.name}"), aug)

    # TL Class Balancing
    if tl_config["balancing_dataset"]:
        current_good = len(list(tl_train_good_dir.glob("*")))
        current_reject = len(list(tl_train_reject_dir.glob("*")))
        if current_reject < current_good:
            diff = current_good - current_reject
            for i in range(diff):
                base_path, cat = random.choice(tl_train_defect)
                aug = apply_dynamic_augmentation(base_path, config)
                if aug is not None:
                    cv2.imwrite(str(tl_train_reject_dir / f"bal_{i}_{cat}_{base_path.name}"), aug)

    # Final Summary (counting actual files on disk)
    print("\n" + "="*55)
    print(f"{'FINAL DATASET SUMMARY (FILES ON DISK)':^55}")
    print("="*55)
    print(f"{'Set Type':<15} | {'Good':<10} | {'Defects':<10}")
    print("-" * 55)
    print(f"{'AD Train':<15} | {len(list(ad_train_good_dir.glob('*'))):<10} | {'-':<10}")
    print(f"{'AD Test':<15} | {len(list(ad_test_good_dir.glob('*'))):<10} | {len(list(ad_test_reject_dir.glob('*'))):<10}")
    print("-" * 55)
    print(f"{'TL Train':<15} | {len(list(tl_train_good_dir.glob('*'))):<10} | {len(list(tl_train_reject_dir.glob('*'))):<10}")
    print(f"{'TL Val':<15} | {len(list(tl_val_good_dir.glob('*'))):<10} | {len(list(tl_val_reject_dir.glob('*'))):<10}")
    print("="*55 + "\n")

if __name__ == "__main__":
    build_mutually_exclusive_datasets()