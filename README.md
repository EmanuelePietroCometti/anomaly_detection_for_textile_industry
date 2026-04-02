# Anomaly Detection for Textile Industry

This repository provides an automated pipeline for anomaly detection in the textile industry. It utilizes state-of-the-art Deep Learning models to identify defects in fabrics and textiles without requiring extensive pixel-level annotations for training.

## Algorithms supported
The project currently supports the following anomaly detection architectures:
1. **PatchCore**: A memory-bank based model that uses a maximal representative memory of nominal patch features. Excellent for rapid adaptation and high performance.
2. **RD4AD (Reverse Distillation for Anomaly Detection)**: A student-teacher architecture where a teacher model distills knowledge into a student model over normal data. Anomalies are detected by measuring the discrepancy between their feature representations.
3. **EAD (Efficient Anomaly Detection)**: An optimized architecture aiming at fast inference while keeping high detection rates.

## Configuration (`config.yaml`)
All project settings are centralized in the `config.yaml` file. You can customize:
* **`general_configuration`**: Global settings like image size (`image_size`), crop size, and valid file extensions (e.g., `.bmp`, `.png`).
* **`datamodule_configuration`**: Dataset paths (root directory, train/test paths, and optionally `mask_dir` for segmentation tasks) and batch sizes.
* **`model_architecture`**: Choose the backbone (e.g., `resnet18`, `wide_resnet50_2`) and the feature extraction layers.
* **Algorithm-specific settings**: Configurations dedicated to specific models, such as `num_epochs` for RD4AD or `coreset_sampling_ratio` for Patchcore.

## How to launch the training
To run the full pipeline (data loading, model training, testing, and metric extraction), use the main entry point:

```bash
# Ensure you have activated your virtual environment (Poetry/Pip)
python main.py --baseline <model_baseline>
```
### Command-Line Arguments

The script exposes 5 command-line arguments to configure and manage the pipeline execution:

* **`--create-dataset`**
    * **Type:** Boolean flag (default: `False`).
    * **Description:** Forces the creation of mutually exclusive datasets prior to training by executing `build_mutually_exclusive_datasets()`.

* **`--baseline`**
    * **Type:** String (default: `"none"`).
    * **Description:** Selects the anomaly detection model to run. Valid choices are `"efficientad"`, `"patchcore"`, or `"rd4ad"`.

* **`--run-transfer-learning`**
    * **Type:** Boolean flag (default: `False`).
    * **Description:** Triggers the transfer learning process to fine-tune the backbone before starting the main anomaly detection pipeline.

* **`--timestamp`**
    * **Type:** String (default: `None`, dynamically generated at runtime).
    * **Description:** Allows manual input of a specific timestamp. This is used to resume a previous training session or load specific custom weights that match the given timestamp.

* **`--exploratory-data-analysis`**
    * **Type:** Boolean flag (default: `False`).
    * **Description:** Runs the exploratory data analysis (EDA) via `apply_eda_analysis()` before starting the training process.

### Configuration guidelines & Model switching

When switching from one model to another (e.g., from Patchcore to RD4AD) or generating new data, it is crucial to correctly update the `config.yaml` file. Follow these three main rules:

1. **Path Management (`paths`)**: The save paths must be checked and updated **every time** you intend to change the type of model to be executed. This prevents the accidental overwriting of reports, anomaly images, checkpoints, and exported models (ONNX) generated from previous runs.
2. **Model-Specific Settings**: It is not necessary to modify the entire file. You should **exclusively** check and adapt the section related to the target model you have chosen to train.
3. **Dataset Versioning**: Every time the dataset is recreated (e.g., by calling the `--create-dataset` argument), it is mandatory to change the version by updating the `dataset_version` parameter.
