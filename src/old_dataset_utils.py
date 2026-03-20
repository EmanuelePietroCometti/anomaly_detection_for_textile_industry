import os
import albumentations as A
import cv2
from pathlib import Path
import shutil
import random
import numpy as np
from config import load_config

def augmentation_dust_threads(source, destination, num_variants=1):
    """
    Function that applies a series of augmentations to the input images, creating augmented versions of the original images.
    - source: the directory containing the original images.
    - destination: the directory where the augmented images will be saved.
    - num_variants: number of augmented variants to create for each original image.
    """
    config = load_config()

    
    input_path = Path(source)
    output_path = Path(destination)
    print(f"Reading from: {input_path.resolve()}")

    output_path.mkdir(parents=True, exist_ok=True)
    
    transform = A.Compose([
        A.HorizontalFlip(p=config["augmentation_params"]["prob_h_flip_augmentation_dust"]),
        A.VerticalFlip(p=config["augmentation_params"]["prob_v_flip_augmentation_dust"]),
    ])

    imgs = list(input_path.glob("*.bmp"))
    valid_extension = config["general_configuration"]["valid_extensions"]
    imgs = [img for img in imgs if img.suffix.lower() in valid_extension]
    print(f"Found {len(imgs)} images with dust and threads!")
    counter_total = 0

    for img_path in imgs:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        basename = img_path.stem
        extension = img_path.suffix
        

        cv2.imwrite(str(output_path / f"{basename}{extension}"), image)
        counter_total += 1 

        for i in range(num_variants):
            augmented = transform(image = image_rgb)
            img_augmentated_rgb = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path / f"{basename}_aug_{i+1}{extension}"), img_augmentated_rgb)
            counter_total += 1
    print(f"Finish! Generated {counter_total} images with dust and threads!")

def augmentation_cimossa(source, destination, num_variants=1):
    """
    Function that applies a horizontal flip and a vertical flip to the input images, creating augmented versions of the original images.
    - source: the directory containing the original images.
    - destination: the directory where the augmented images will be saved.
    - num_variants: number of augmented variants to create for each original image. For example, if num_variants is set to 2, the function will create two augmented versions of each original image, 
    in addition to the original image itself.
    """
    config = load_config()
    
    input_path = Path(source)
    output_path = Path(destination)
    print(f"Reading from: {input_path.resolve()}")


    output_path.mkdir(parents=True, exist_ok=True)
    
    transform = A.Compose([
        A.HorizontalFlip(p=config["augmentation_params"]["prob_h_flip_augmentation_cimossa"]),
        A.VerticalFlip(p=config["augmentation_params"]["prob_v_flip_augmentation_cimossa"])
    ])

    imgs = list(input_path.glob("*.bmp"))
    valid_extension = config["general_configuration"]["valid_extensions"]
    imgs = [img for img in imgs if img.suffix.lower() in valid_extension]
    print(f"Found {len(imgs)} images with cimossa!")
    counter_total = 0

    for img_path in imgs:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        basename = img_path.stem
        extension = img_path.suffix
        

        cv2.imwrite(str(output_path / f"{basename}{extension}"), image)
        counter_total += 1 

        for i in range(num_variants):
            augmented = transform(image = image_rgb)
            img_augmentated_rgb = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path / f"{basename}_aug_{i+1}{extension}"), img_augmentated_rgb)
            counter_total += 1
    print(f"Finish! Generated {counter_total} images with cimossa!")

def stretch_textile_horizontal(image_path, destination, force=0.0, reverse_light=False, intensity=0.4, phase=0.0, num_waves=1.0):
    """
    Applies a deformation along the X-axis creating VERTICAL folds.
    Includes both stretching and the illumination map (highlights and shadows).
    """
    image_path = Path(image_path)
    basename = image_path.stem
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    H, W = image.shape[:2]

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # Deformation along the X-axis
    frequency = X / W * np.pi * num_waves + phase
    X_deformed = X + force * np.sin(frequency)
    
    # Illumination calculation
    direction = -1.0 if reverse_light else 1.0
    inclination = np.cos(frequency) * direction
    map_light = 1.0 + (intensity * inclination)
    
    os.makedirs(destination, exist_ok=True)
    
    # Remap for image deformation
    image_distorted = cv2.remap(
        image, X_deformed, Y, 
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    
    # Remap to align the light map with the deformation
    map_light_distortion = cv2.remap(
        map_light.astype(np.float32), X_deformed, Y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    # Applying the light
    light_distorted = image_distorted.astype(np.float32) * map_light_distortion[..., np.newaxis]
    img_illuminated = np.clip(light_distorted, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(destination, f"{basename}_illuminated_{phase}_{num_waves}_h.bmp"), img_illuminated)
    
def stretch_textile_vertical(image_path, destination, force=0.0, phase=0.0, num_waves=1.0):
    """
    Applies a deformation along the Y-axis creating HORIZONTAL folds.
    Performs ONLY stretching, without applying light variations.
    """
    image_path = Path(image_path)
    basename = image_path.stem
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    H, W = image.shape[:2]

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # Deformation along the Y-axis
    frequency = Y / H * np.pi * num_waves + phase
    Y_deformed = Y + force * np.sin(frequency)
    
    os.makedirs(destination, exist_ok=True)
    
    # Remap of the image only
    image_distorted = cv2.remap(
        image, X, Y_deformed, 
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    
    # Direct save (note the suffix changed to _stretched_)
    cv2.imwrite(os.path.join(destination, f"{basename}_stretched_{phase}_{num_waves}_v.bmp"), image_distorted)

def augmentation_textile(source, destination, num_variants=1):
    """
    Function that applies a realistic deformation that moves the threads (dilates) in one area
    and brings them closer (compresses) in another. 
    - num_variants: number of augmented variants to create for each original image. For example, if num_variants is set to 2, the function will create two augmented versions of each original image, in addition to the original image itself."""
    config = load_config()
    
    input_path = Path(source)
    output_path = Path(destination)
    print(f"Reading from: {input_path.resolve()}")

    output_path.mkdir(parents=True, exist_ok=True)

    imgs = list(input_path.glob("*.bmp"))
    valid_extension = config["general_configuration"]["valid_extensions"]
    imgs = [img for img in imgs if img.suffix.lower() in valid_extension]
    print(f"Found {len(imgs)} images with textile defects!")
    counter_total = 0
    while counter_total < num_variants * len(imgs):
        for img_path in imgs:
            intensity = random.uniform(config["augmentation_params"]["intensity_min_range_augmentation_textile"], config["augmentation_params"]["intensity_max_range_augmentation_textile"])
            phase = random.uniform(0, np.pi)
            num_waves = random.uniform(config["augmentation_params"]["min_num_waves_augmentation_textile"],config["augmentation_params"]["max_num_waves_augmentation_textile"])
            force = random.uniform(config["augmentation_params"]["min_force_augmentation_textile"], config["augmentation_params"]["max_force_augmentation_textile"])
            reverse_light = random.choice([True, False])
            stretch_textile_horizontal(img_path, output_path, force=force, reverse_light=reverse_light, intensity=intensity, phase=phase, num_waves=num_waves)
            stretch_textile_vertical(img_path, output_path,force=force, phase=phase, num_waves=num_waves)
            counter_total += 2
    print(f"Finish! Generated {counter_total} deformed images!")

def select_random_images(source_folder, destination_folder_train, destination_folder_test, percentage=0.10):
    """
    Function that selects a random percentage of images from the source folder and copies them to the destination folder.
    - source_folder: the directory containing the original images.
    - destination_folder_train: the directory where the selected training images will be copied.
    - destination_folder_test: the directory where the selected test images will be copied.
    - percentage: the percentage of images to be selected and copied (e.g., 0.10 for 10%).
    """
    source_path = Path(source_folder)
    destination_path_train = Path(destination_folder_train)
    destination_path_test = Path(destination_folder_test)

    destination_path_train.mkdir(parents=True, exist_ok=True)
    destination_path_test.mkdir(parents=True, exist_ok=True)

    valid_extension = ['.bmp']
    imgs = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extension]

    num_images_to_select = int(len(imgs) * percentage)
    selected_images = random.sample(imgs, num_images_to_select)

    for img in selected_images:
        shutil.copy2(img, destination_path_train / img.name)

    print(f"Selected {num_images_to_select} images from {source_folder} and copied them to {destination_folder_train}.")  

def split_dataset_good(source_folder, destination_folder, percentage_training=0.35, percentage_test=0.10):
    """
    Function that splits the dataset of "good" images into training and test sets, maintaining the original folder structure.
    - source_folder: the root directory containing the original dataset, organized in subfolders for each
    - destination_folder: the root directory where the new training and test folders will be created.
    - percentage_training: the percentage of images to be allocated to the training set (e.g., 0.35 for 35%).
    - percentage_test: the percentage of images to be allocated to the test set (e.g., 0.10 for 10%).
    """
    original_path = Path(source_folder)
    
    dir_training = Path(destination_folder) / 'train'   
    dir_test_good = Path(destination_folder) / 'test' / 'good'

    dir_training.mkdir(parents=True, exist_ok=True)
    dir_test_good.mkdir(parents=True, exist_ok=True)

    config = load_config()

    valid_extension = config["general_configuration"]["valid_extensions"]

    
    for sensor_dir in original_path.iterdir():
        if sensor_dir.is_dir():
            dir_ok = sensor_dir / 'OK'

            if dir_ok.exists() and dir_ok.is_dir():
                imgs = [
                    f for f in dir_ok.iterdir() 
                    if f.is_file() and f.suffix.lower() in valid_extension
                ]

                if not imgs:
                    continue

    
                random.shuffle(imgs)

        
                total_imgs = len(imgs)
                qty_training = int(total_imgs * percentage_training)
                qty_test = int(total_imgs * percentage_test)

                
                imgs_training = imgs[:qty_training]
                imgs_test = imgs[qty_training:qty_training + qty_test]

               
                for img in imgs_training:
                    shutil.copy2(img, dir_training / img.name)


                for img in imgs_test:
                    shutil.copy2(img, dir_test_good / img.name)

                print(f"{sensor_dir.name}: {total_imgs} total OK images -> {len(imgs_training)} in training, {len(imgs_test)} in test/good.")

    print("\nOperation completed successfully!")

def split_dataset(source_folder, train_folder, test_folder, percentage_training=0.5, percentage_test=0.05):
    """
    Function that splits the dataset passed as input into training and test sets, maintaining the original folder structure.
    - source_folder: the root directory containing the original dataset, organized in subfolders for each
    - train_folder: the directory where the training images will be saved.
    - test_folder: the directory where the test images will be saved.
    - percentage_training: the percentage of images to be allocated to the training set (e.g 0.5 for 50%).
    - percentage_test: the percentage of images to be allocated to the test set (e.g. 0.05 for 5%).
    """
    config = load_config()

    dir_source = Path(source_folder)
    
    dir_train = Path(train_folder)
    dir_test = Path(test_folder)
    
    dir_train.mkdir(parents=True, exist_ok=True)
    dir_test.mkdir(parents=True, exist_ok=True)

    valid_extension = config["general_configuration"]["valid_extensions"]

    if dir_source.exists() and dir_source.is_dir():
        imgs = [
            f for f in dir_source.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_extension
        ]

        random.shuffle(imgs)


        total_imgs = len(imgs)
        qty_training = int(total_imgs * percentage_training)
        qty_test = int(total_imgs * percentage_test)

        imgs_training = imgs[:qty_training]
        imgs_test = imgs[qty_training:qty_training + qty_test]

        for img in imgs_training:
            shutil.copy2(img, dir_train / img.name)

        for img in imgs_test:
            shutil.copy2(img, dir_test / img.name)

        print(f"Finished splitting dataset! Total images: {total_imgs}, moved to test: {len(imgs_test)} and {len(imgs_training)} moved to train.")

def extract_reject_images(destination_folder, percentage=0.1):
    """
    Extracts "reject" images recursively from a LIST of source directories 
    and saves them ALL into a SINGLE flat destination folder.
    
    To prevent overwriting files with the same name from different subfolders, 
    it prefixes the parent folder's name to the image filename during the copy process.
    
    Args:
        destination_folder (str or Path): The single output directory for all rejected images.
        percentage (float): The fraction of total reject images to extract (default is 0.1 for 10%).
    """
    config = load_config()
    source_folders_nok = config["paths"]["source_folder_nok"]
    valid_extension = config["general_configuration"]["valid_extensions"]

    if isinstance(source_folders_nok, str):
        source_folders_nok = [source_folders_nok]

    dir_destination_reject = Path(destination_folder)
    dir_destination_reject.mkdir(parents=True, exist_ok=True)
    
    imgs = []
    for folder_path in source_folders_nok:
        source_root = Path(folder_path)
        if not source_root.exists():
            print(f"Warning: The folder {source_root} does not exist. Skipping...")
            continue
         
        imgs.extend([
            f for f in source_root.rglob('*') 
            if f.is_file() and f.suffix.lower() in valid_extension
        ])

    if not imgs:
        print("No REJECT images found in the specified source folders.")
        return

    random.shuffle(imgs)
    num_images_to_extract = int(len(imgs) * percentage)
    selected_images = imgs[:num_images_to_extract]

    for img in selected_images:
        parent_folder_name = img.parent.name
        new_filename = f"{parent_folder_name}_{img.name}"
        
        dest_path = dir_destination_reject / new_filename
        while dest_path.exists():
            random_number = random.randint(1000, 9999)
            new_filename = f"{parent_folder_name}_{random_number}_{img.name}"
            dest_path = dir_destination_reject / new_filename

        shutil.copy2(img, dest_path)

    print(f"Successfully extracted {len(selected_images)} REJECT images (out of {len(imgs)} total found).")
    print(f"All images have been saved to the flat directory: {dir_destination_reject}")

def create_dataset():
    config = load_config()
    
    split_dataset_good(config["paths"]["source_good_images"], config["paths"]["destination_good_images"], config["split_ratios"]["train_ratio"], config["split_ratios"]["test_ratio_good"])
    split_dataset(config["paths"]["source_cimossa"], config["paths"]["destination_training_cimossa"], config["paths"]["destination_test_cimossa"], config["split_ratios"]["split_ratio_cimossa_train"], config["split_ratios"]["split_ratio_cimossa_test"])
    augmentation_cimossa(config["paths"]["destination_training_cimossa"], config["paths"]["destination_cimossa_augmentation_train"], num_variants=config["split_ratios"]["num_variants_augmentation_cimossa_train"])
    augmentation_cimossa(config["paths"]["destination_test_cimossa"], config["paths"]["destination_cimossa_augmentation_test"], num_variants=config["split_ratios"]["num_variants_augmentation_cimossa_test"])
    split_dataset(config["paths"]["source_folder_textile"], config["paths"]["destination_folder_textile_original_train"],  config["paths"]["destination_folder_textile_original_test"], config["split_ratios"]["split_ratio_textile_train"], config["split_ratios"]["split_ratio_textile_test"])
    augmentation_textile(config["paths"]["destination_folder_textile_original_train"], config["paths"]["destination_folder_textile_augmented_train"],num_variants=config["split_ratios"]["num_variants_textile_augmentation"])
    augmentation_textile(config["paths"]["destination_folder_textile_original_test"], config["paths"]["destination_folder_textile_augmented_test"],num_variants=config["split_ratios"]["num_variants_textile_augmentation"])
    split_dataset(config["paths"]["source_dust_images"], config["paths"]["destination_dust_images_train"],  config["paths"]["destination_dust_images_test"], config["split_ratios"]["split_ratio_dust_train"], config["split_ratios"]["split_ratio_dust_test"])
    augmentation_dust_threads(config["paths"]["destination_dust_images_train"], config["paths"]["destination_dust_images_augmentation_train"], num_variants=config["split_ratios"]["num_variants_augmentation_dust_threads_train"]) 
    augmentation_dust_threads(config["paths"]["destination_dust_images_test"], config["paths"]["destination_dust_images_augmentation_test"], num_variants=config["split_ratios"]["num_variants_augmentation_dust_threads_test"]) 
    extract_reject_images(config["paths"]["destination_reject_images"], 1.0)
    
    valid_extension = config["general_configuration"]["valid_extensions"]
    counter_test_total = 0
    counter_train_total = 0
    
    for source in config["paths"]["paths_dataset"]:
        dir_train = Path(source) / config["paths"]["folder_train"]
        dir_test = Path(source) / config["paths"]["folder_test"] 
        
        dir_destination_train = Path(config["paths"]["train_path"])
        dir_destination_test = Path(config["paths"]["test_good_path"])
        
        dir_destination_train.mkdir(parents=True, exist_ok=True)
        dir_destination_test.mkdir(parents=True, exist_ok=True)
        
        for file in dir_train.iterdir():
            if file.is_file() and file.suffix.lower() in valid_extension:
                shutil.copy2(file, dir_destination_train / file.name)
                counter_train_total += 1

        for file in dir_test.iterdir():
            if file.is_file() and file.suffix.lower() in valid_extension:
                shutil.copy2(file, dir_destination_test / file.name)
                counter_test_total += 1

    print(f"Total images copied in train sets: {counter_train_total}")
    print(f"Total images copied in test sets: {counter_test_total}")

if __name__ == "__main__":
    create_dataset()