import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
from .config import load_config
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from pathlib import Path
from sklearn.metrics import roc_curve, auc as sklearn_auc



def extract_features(folder_map, config):
    """
    Extracts features from images in the specified folders using a pre-trained ResNet18 model. The function processes each image, applies necessary transformations, and collects the resulting feature vectors along with their corresponding labels and file paths. It handles potential issues such as missing folders or loading errors gracefully, providing informative warnings to the user.
    Args:
        folder_map (dict): A dictionary mapping folder paths to their corresponding labels (e.g., {"./data/train": "Train - Good", "./data/test/good": "Test - Good", "./data/test/reject": "Test - Defect"}).
        config (dict): The configuration dictionary loaded from the YAML file, which may contain parameters for image transformations and valid file extensions.
    Returns:
        features (numpy.ndarray): A matrix of extracted features where each row corresponds to an image.
        labels (numpy.ndarray): The array of labels corresponding to each feature vector.
        image_paths (numpy.ndarray): The array of file paths corresponding to each feature vector.

    """
    
    config = load_config()
    model_config = config.get("model_architecture", {})
    gen_config = config.get("general_configuration", {})
    try:    
        model_name = model_config.get("backbone", "resnet18")
        weights_name = model_config.get("weights", "DEFAULT")
        model_builder = getattr(models, model_name)
        model = model_builder(weights = weights_name)
    except:
        return ValueError(f"Model {model_name} is not  found in torchvision.models. Check PyTorch documentation and insert a valid name!")
    
    if hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()
    elif hasattr(model, 'classifier'):
        model.classifier = torch.nn.Identity()
    elif hasattr(model, 'head'):
        model.head = torch.nn.Identity()
    else:
        print("Warning: Could not automatically detect the classification head. Extracted features might still be logits.")
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read parameters from config.yaml
    gen_config = config.get("general_configuration", {})
    img_size = gen_config.get("image_size", [512, 512])
    
    # Ensure valid_extensions is a tuple (required by string.endswith())
    valid_extensions = tuple(gen_config.get("valid_extensions", [".bmp", ".BMP"]))

    transform = transforms.Compose([
        transforms.v2.Resize(tuple(img_size)),
        transforms.v2.ToImage(),
        transforms.v2.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    features = []
    labels = []
    image_paths = []
    
    print("Extracting features...")
    with torch.no_grad():
        for folder_path, label_name in folder_map.items():
            if not os.path.exists(folder_path):
                print(f"Warning: Folder '{folder_path}' does not exist. Skipping.")
                continue
                
            print(f"-> Analyzing: {folder_path} ({label_name})")
            
            for file_name in os.listdir(folder_path):
                if not file_name.lower().endswith(valid_extensions):
                    continue
                    
                img_path = os.path.join(folder_path, file_name)
                
                if os.path.isfile(img_path):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_t = transform(img).unsqueeze(0).to(device)
                        
                        feat = model(img_t).cpu().numpy().flatten()
                        features.append(feat)
                        labels.append(label_name)
                        image_paths.append(img_path)
                    except Exception as e:
                        print(f"Loading error for {file_name}: {e}")

    return np.array(features), np.array(labels), np.array(image_paths)

def analyze_pca_variance(features_matrix, destination_dir="results/EDA"):
    """
    Analyze the explained variance of PCA components to determine how many dimensions are needed to capture most of the variance.
    Args:
        features_matrix (numpy.ndarray): The matrix of extracted features (samples x features).
        destination_path (str): The path where the diagnostic plot will be saved.
    """
    config = load_config()
    dataset_cfg = config.get("dataset_pipeline")
    dataset_version = dataset_cfg.get("dataset_version")
    destination_path = Path(destination_dir) / dataset_version / "plot_PCA_variance.png"


    print("\n --- PCA VARIANCE ANALYSIS ---")
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_matrix)

    print("Computing Principal Components...")
    pca = PCA()
    pca.fit(X_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    n_90 = np.argmax(cumulative_variance >= 0.90) + 1
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_99 = np.argmax(cumulative_variance >= 0.99) + 1

    print("\n" + "="*50)
    print("DIMENSIONALITY ANALYSIS RESULTS (PCA)")
    print("="*50)
    print(f"To retain 90% of the information, you only need: {n_90} components")
    print(f"To retain 95% of the information, you only need: {n_95} components")
    print(f"To retain 99% of the information, you only need: {n_99} components")
    print(f"The remaining {features_matrix.shape[1] - n_99} components are ALMOST CERTAINLY NOISE.")
    print("="*50 + "\n")

    # Generate Diagnostic Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, linewidth=2, color='#1f77b4')
    
    # Horizontal and vertical threshold lines
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label=f'95% Variance ({n_95} comp.)')
    plt.axvline(x=n_95, color='red', linestyle=':', alpha=0.7)
    
    plt.axhline(y=0.99, color='green', linestyle='--', alpha=0.7, label=f'99% Variance ({n_99} comp.)')
    plt.axvline(x=n_99, color='green', linestyle=':', alpha=0.7)

    plt.title('Cumulative Explained Variance (Cumulative Scree Plot)', fontsize=14, pad=15)
    plt.xlabel('Number of Principal Components (PCA)', fontsize=12)
    plt.ylabel('Cumulative Explained Variance Ratio (0.0 - 1.0)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save and display the plot
    plt.tight_layout()
    plt.savefig(destination_path, dpi=300)
    print(f"Plot saved as '{destination_path}'")
    plt.show()
    
    return n_95

def analyze_pca_feature_importance(features_good, features_defect, n_components, destination_dir="results/EDA"):
    """
    Applies the Kolmogorov-Smirnov test to each PCA component to determine 
    which components are most statistically significant for separating Good vs Defect.
    """
    config = load_config()
    dataset_cfg = config.get("dataset_pipeline")
    dataset_version = dataset_cfg.get("dataset_version")
    destination_path = Path(destination_dir) / dataset_version
    print("\n--- PCA COMPONENT SEPARABILITY (KS-TEST) ---")
    
    # Combine and reduce
    X = np.vstack((features_good, features_defect))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    good_pca = X_pca[:len(features_good)]
    defect_pca = X_pca[len(features_good):]
    
    # Run KS-Test for each component
    ks_results = []
    for i in range(n_components):
        stat, p_value = ks_2samp(good_pca[:, i], defect_pca[:, i])
        ks_results.append({'Component': i+1, 'KS_Statistic': stat, 'P_Value': p_value})
        
    df_ks = pd.DataFrame(ks_results).sort_values(by='KS_Statistic', ascending=False)
    
    print("Top 5 most discriminative PCA components:")
    print(df_ks.head(5).to_string(index=False))
    
    destination_path = destination_path / "PCA_feature_importance.csv"
    df_ks.to_csv(destination_path, index=False)
    # A high KS Statistic (closer to 1.0) means the distributions are very different
    return df_ks


def analyze_mahalanobis_distance(features_train_good, features_test_good, features_test_defect, n_pca_components=50, destination_dir="results/EDA"):
    """
    Calculates the Mahalanobis distance of test samples from the training 'Good' distribution.
    This is a core metric for Anomaly Detection to check if defects are statistical outliers.
    """
    config = load_config()
    dataset_cfg = config.get("dataset_pipeline")
    dataset_version = dataset_cfg.get("dataset_version")
    destination_path = Path(destination_dir) / dataset_version / "plot_Mahalanobis_distance.png"
    print("\n--- MAHALANOBIS DISTANCE ANALYSIS ---")
    
    # Reduce dimensionality to avoid singular matrix errors
    pca = PCA(n_components=n_pca_components)
    train_good_pca = pca.fit_transform(features_train_good)
    test_good_pca = pca.transform(features_test_good)
    test_defect_pca = pca.transform(features_test_defect)
    
    # Compute mean and inverse covariance matrix of the GOOD training set
    centroid_good = np.mean(train_good_pca, axis=0)
    cov_matrix = np.cov(train_good_pca, rowvar=False)
    inv_cov_matrix = inv(cov_matrix)
    
    # Calculate distances
    dist_test_good = [mahalanobis(x, centroid_good, inv_cov_matrix) for x in test_good_pca]
    dist_test_defect = [mahalanobis(x, centroid_good, inv_cov_matrix) for x in test_defect_pca]
    
    # Plot distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(dist_test_good, color="green", label="Test Good", kde=True, stat="density", alpha=0.5)
    sns.histplot(dist_test_defect, color="red", label="Test Defect", kde=True, stat="density", alpha=0.5)
    plt.title("Mahalanobis Distance from 'Good' Distribution", fontsize=14)
    plt.xlabel("Distance Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(destination_path, dpi=300)
    print(f"Plot saved as '{destination_path}'")
    plt.show()
    
    # Calculate separability overlap (using a simple threshold based on max Good distance)
    threshold = np.percentile(dist_test_good, 95) # 95th percentile of normal data
    detected_defects = sum(d > threshold for d in dist_test_defect) / len(dist_test_defect)
    print(f"Separability estimation (Threshold at 95th percentile of Good): {detected_defects*100:.2f}% defects detected.")

def evaluate_unsupervised_baseline(features_train_good, features_test_good, features_test_defect, n_pca_components=50, destination_dir="results/EDA"):
    """
    Trains an Isolation Forest strictly on 'Good' data to see if defects 
    can be detected purely as outliers in the feature space.
    """
    config = load_config()
    dataset_cfg = config.get("dataset_pipeline", {})
    dataset_version = dataset_cfg.get("dataset_version", "default")
    
    # Path Management (Robust)
    output_dir = Path(destination_dir) / dataset_version
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    plot_path = output_dir / "plot_auroc_isolation_forest.png"

    print("\n--- UNSUPERVISED BASELINE (ISOLATION FOREST) ---")
    
    # Dimensionality Reduction
    pca = PCA(n_components=n_pca_components)
    train_good_pca = pca.fit_transform(features_train_good)
    test_good_pca = pca.transform(features_test_good)
    test_defect_pca = pca.transform(features_test_defect)
    
    # Model Training
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(train_good_pca)
    
    # Scoring
    scores_good = -iso_forest.score_samples(test_good_pca)
    scores_defect = -iso_forest.score_samples(test_defect_pca)
    
    y_true = np.hstack([np.zeros(len(scores_good)), np.ones(len(scores_defect))])
    y_scores = np.hstack([scores_good, scores_defect])
    
    # Metrics Calculation (Fixed variable naming)
    current_auc_score = roc_auc_score(y_true, y_scores)
    print(f"Isolation Forest Baseline AUROC: {current_auc_score:.4f}")
    
    # ROC Plotting
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc_val = sklearn_auc(fpr, tpr) # Use the renamed sklearn function

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Isolation Forest ROC (AUC = {roc_auc_val:.4f})')
    plt.fill_between(fpr, tpr, alpha=0.15, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Unsupervised Baseline: Isolation Forest\nDataset Version: {dataset_version}')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Baseline plot saved at: {plot_path}")
    plt.close()
    
    return current_auc_score

def plot_interactive_tsne(features, labels, image_paths, n_components_pca=50, destination_dir="results/EDA"):
    """
    Plots an interactive t-SNE visualization of the feature space. After reducing dimensionality with PCA, it applies t-SNE to project the features into 2D. Each point is colored by its label, and clicking on a point will open the corresponding image for visual inspection.
    Args:
    - features (numpy.ndarray): The matrix of extracted features (samples x features).
    - labels (numpy.ndarray): The array of labels corresponding to each feature vector.
    - image_paths (numpy.ndarray): The array of file paths corresponding to each feature vector.
    - n_components_pca (int): The number of PCA components to retain before applying t-SNE (default: 50).
    - destination_path (str): The path where the interactive plot will be saved (default: "interactive_tsne_plot.png").
    """
    config = load_config()
    dataset_cfg = config.get("dataset_pipeline")
    dataset_version = dataset_cfg.get("dataset_version")
    destination_path = Path(destination_dir) / dataset_version / "interactive_tsne_plot.png"
    print("Dimensionality reduction with PCA + t-SNE...")
    
    pca = PCA(n_components=n_components_pca)
    features_pca = pca.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features_pca)
    
    unique_labels = np.unique(labels)
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    numeric_labels = [label_to_int[lbl] for lbl in labels]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=numeric_labels, 
        cmap='Set1',
        alpha=0.7, 
        s=100, 
        picker=5 
    )
    
    handles, _ = scatter.legend_elements()
    ax.legend(handles, unique_labels, title="Dataset")
    
    plt.title("Interactive Analysis: Click on a point to view the relative image!")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    def on_pick(event):
        ind = event.ind[0] 
        clicked_image_path = image_paths[ind]
        clicked_label = labels[ind]
        
        print(f"\n[{clicked_label}] -> Opening file: {clicked_image_path}")
        Image.open(clicked_image_path).show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    print("\nPlot ready! Use the mouse to explore the points.")
    plt.savefig(destination_path, dpi=300)
    print(f"Plot saved as '{destination_path}'")
    plt.show()

def realistic_pca_lda_analysis(features_good, features_defect, n_pca_components, destination_dir="results/EDA"):
    """
    Executes a dimensionality reduction pipeline (PCA) followed by classification (LDA)
    to obtain a realistic estimate of defect separability.
    
    :param features_good: Numpy array of features extracted from normal (good) images.
    :param features_defect: Numpy array of features extracted from defective images.
    :param n_pca_components: The number obtained from the variance plot (e.g., 95% cut-off).
    :param destination_path: The path where the analysis plot will be saved.
    """
    config = load_config()
    dataset_cfg = config.get("dataset_pipeline")
    dataset_version = dataset_cfg.get("dataset_version")
    destination_path = Path(destination_dir) / dataset_version / "pca_lda_analysis.png"
    print(f"Input data: {len(features_good)} Good, {len(features_defect)} Defects.")
    
    # Labels (0 = Good, 1 = Defect)
    X = np.vstack((features_good, features_defect))
    y = np.hstack((np.zeros(len(features_good)), np.ones(len(features_defect))))
    
    # Standardization
    print("Standardizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality Reduction (PCA)
    print(f"Applying PCA: Compressing from {X.shape[1]} to {n_pca_components} dimensions...")
    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Train LDA on the TRUE signal
    print("Training LDA on the reduced space...")
    lda = LDA()
    X_lda = lda.fit_transform(X_pca, y)
    
    y_pred = lda.predict(X_pca)
    acc = accuracy_score(y, y_pred)
    print("\n" + "="*50)
    print(f"REAL LINEAR ACCURACY: {acc*100:.2f}%")
    print("="*50 + "\n")
    
    plt.figure(figsize=(10, 6))
    
    lda_good = X_lda[y == 0].flatten()
    lda_defect = X_lda[y == 1].flatten()
    
    # Plot distributions with KDE (Kernel Density Estimate)
    sns.histplot(lda_good, color="#1f77b4", label="Good (0)", kde=True, stat="density", bins=30, alpha=0.5)
    sns.histplot(lda_defect, color="#d62728", label="Defect (1)", kde=True, stat="density", bins=30, alpha=0.5)
    
    plt.title(f'Realistic LDA Projection (on {n_pca_components} PCA Components)\nAccuracy: {acc*100:.2f}%', fontsize=14, pad=15)
    plt.xlabel('LDA Discriminant Axis', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(destination_path, dpi=300)
    print(f"Plot saved as '{destination_path}'")
    plt.show()

def apply_eda_analysis():
    config = load_config("config.yaml")
    
    datamodule_cfg = config.get("datamodule_configruation", {})
    train_dir = datamodule_cfg.get("train_dir", "./data/dataset_patchcore/train")
    test_good_dir = datamodule_cfg.get("test_dir_good", "./data/dataset_patchcore/test/good")
    test_defect_dir = datamodule_cfg.get("test_dir_reject", "./data/dataset_patchcore/test/reject")
    paths = config.get("paths", {})
    eda_dir = paths.get("eda_path", "results/EDA")

    dataset_cfg = config.get("dataset_pipeline",  "results/EDA")
    dataset_version = dataset_cfg.get("dataset_version", "v1.0.0")

    version_path = Path(eda_dir)/dataset_version
    version_path.mkdir(parents=True, exist_ok=True)

    folder_map = {
        train_dir: "Train - Good",
        test_good_dir: "Test - Good",
        test_defect_dir: "Test - Defect"
    }
    
    # Feature Extraction
    X, Y, paths = extract_features(folder_map, config) 
    
    if len(X) > 0:
        print("\n" + "="*50)
        print("STARTING ANOMALY DETECTION EDA PIPELINE")
        print("="*50)

        # Separate the dataset into specific sets based on labels
        # This is crucial for Anomaly Detection where we train on Good and test on Good+Defect
        train_good = X[Y == "Train - Good"]
        test_good = X[Y == "Test - Good"]
        test_defect = X[Y == "Test - Defect"]
        
        paths_config = config.get("paths", {})

        # Dimensionality Assessment
        # We run this on all data (X) to understand the global variance
        n_95 = analyze_pca_variance(X, destination_dir=eda_dir)
        # Ensure we have the necessary classes before running statistical tests
        if len(train_good) > 0 and len(test_defect) > 0:
            
            # Feature Importance (KS-Test)
            # Compare normal training data vs defects to find discriminative components
            analyze_pca_feature_importance(train_good, test_defect, n_components=n_95, destination_dir=eda_dir)
            
            if len(test_good) > 0:
                # Spatial & Distance Analysis (Mahalanobis)
                # Check if test sets deviate from the training distribution
                analyze_mahalanobis_distance(train_good, test_good, test_defect, n_pca_components=n_95, destination_dir=eda_dir)
                
                # Unsupervised Baseline (Isolation Forest)
                # Get a standard AUROC metric for separability
                evaluate_unsupervised_baseline(train_good, test_good, test_defect, n_pca_components=n_95, destination_dir=eda_dir)
            else:
                print("\n[Warning] 'Test - Good' data missing. Skipping Mahalanobis and Isolation Forest tests.")
                
        else:
            print("\n[Warning] Missing 'Train - Good' or 'Test - Defect' data. Skipping statistical tests.")

        # Visual Explorations (Qualitative checks)
        if len(train_good) > 0 and len(test_defect) > 0:
            realistic_pca_lda_analysis(train_good, test_defect, n_pca_components=n_95, destination_dir=eda_dir)
            
        plot_interactive_tsne(X, Y, paths, n_components_pca=n_95, destination_dir=eda_dir)

    else:
        print("No images found! Check the paths entered in your yaml or script.")


if __name__ == "__main__":
    config = load_config("config.yaml")
    
    datamodule_cfg = config.get("datamodule_configruation", {})

    train_dir = datamodule_cfg.get("train_dir", "./data/dataset_patchcore/train")
    test_good_dir = datamodule_cfg.get("test_dir_good", "./data/dataset_patchcore/test/good")
    test_defect_dir = datamodule_cfg.get("test_dir_reject", "./data/dataset_patchcore/test/reject")
    paths = config.get("paths", {})
    eda_dir = paths.get("eda_path", "results/EDA")

    dataset_cfg = config.get("dataset_pipeline", "results/EDA")
    dataset_version = dataset_cfg.get("dataset_version", "v1.0.0")
    version_path = Path(eda_dir)/dataset_version
    version_path.mkdir(parents=True, exist_ok=True)

    folder_map = {
        train_dir: "Train - Good",
        test_good_dir: "Test - Good",
        test_defect_dir: "Test - Defect"
    }
    
    # Feature Extraction
    X, Y, paths = extract_features(folder_map, config) 
    
    if len(X) > 0:
        print("\n" + "="*50)
        print("STARTING ANOMALY DETECTION EDA PIPELINE")
        print("="*50)

        # Separate the dataset into specific sets based on labels
        # This is crucial for Anomaly Detection where we train on Good and test on Good+Defect
        train_good = X[Y == "Train - Good"]
        test_good = X[Y == "Test - Good"]
        test_defect = X[Y == "Test - Defect"]
        
        paths_config = config.get("paths", {})

        # Dimensionality Assessment
        # We run this on all data (X) to understand the global variance
        n_95 = analyze_pca_variance(X, destination_dir=eda_dir)
        # Ensure we have the necessary classes before running statistical tests
        if len(train_good) > 0 and len(test_defect) > 0:
            
            # Feature Importance (KS-Test)
            # Compare normal training data vs defects to find discriminative components
            analyze_pca_feature_importance(train_good, test_defect, n_components=n_95, destination_dir=eda_dir)
            
            if len(test_good) > 0:
                # Spatial & Distance Analysis (Mahalanobis)
                # Check if test sets deviate from the training distribution
                analyze_mahalanobis_distance(train_good, test_good, test_defect, n_pca_components=n_95, destination_dir=eda_dir)
                
                # Unsupervised Baseline (Isolation Forest)
                # Get a standard AUROC metric for separability
                evaluate_unsupervised_baseline(train_good, test_good, test_defect, n_pca_components=n_95, destination_dir=eda_dir)
            else:
                print("\n[Warning] 'Test - Good' data missing. Skipping Mahalanobis and Isolation Forest tests.")
                
        else:
            print("\n[Warning] Missing 'Train - Good' or 'Test - Defect' data. Skipping statistical tests.")

        # Visual Explorations (Qualitative checks)
        if len(train_good) > 0 and len(test_defect) > 0:
            realistic_pca_lda_analysis(train_good, test_defect, n_pca_components=n_95, destination_dir=eda_dir)
            
        plot_interactive_tsne(X, Y, paths, n_components_pca=n_95, destination_dir=eda_dir)

    else:
        print("No images found! Check the paths entered in your yaml or script.")