import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer,f1_score, balanced_accuracy_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate
import seaborn as sns
from config import load_config
from sklearn.preprocessing import StandardScaler



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
    # Initialize ResNet18 (Feature Extractor only)
    model = models.wide_resnet50_2(weights="DEFAULT")
    model.fc = torch.nn.Identity()
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read parameters from config.yaml
    gen_config = config.get("general_configuration", {})
    img_size = gen_config.get("image_size", [512, 512])
    
    # Ensure valid_extensions is a tuple (required by string.endswith())
    valid_extensions = tuple(gen_config.get("valid_extensions", [".bmp", ".jpg", ".jpeg", ".png"]))

    transform = transforms.Compose([
        transforms.Resize(tuple(img_size)),
        transforms.ToTensor(),
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

def plot_interactive_tsne(features, labels, image_paths, n_components_pca=50, destination_path="interactive_tsne_plot.png"):
    """
    Plots an interactive t-SNE visualization of the feature space. After reducing dimensionality with PCA, it applies t-SNE to project the features into 2D. Each point is colored by its label, and clicking on a point will open the corresponding image for visual inspection.
    Args:
    - features (numpy.ndarray): The matrix of extracted features (samples x features).
    - labels (numpy.ndarray): The array of labels corresponding to each feature vector.
    - image_paths (numpy.ndarray): The array of file paths corresponding to each feature vector.
    - n_components_pca (int): The number of PCA components to retain before applying t-SNE (default: 50).
    - destination_path (str): The path where the interactive plot will be saved (default: "interactive_tsne_plot.png").
    """
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

def realistic_pca_lda_analysis(features_good, features_defect, n_pca_components, destination_path="pca_lda_analysis.png"):
    """
    Executes a dimensionality reduction pipeline (PCA) followed by classification (LDA)
    to obtain a realistic estimate of defect separability.
    
    :param features_good: Numpy array of features extracted from normal (good) images.
    :param features_defect: Numpy array of features extracted from defective images.
    :param n_pca_components: The number obtained from the variance plot (e.g., 95% cut-off).
    :param destination_path: The path where the analysis plot will be saved.
    """
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

def analyze_pca_variance(features_matrix, destination_path="pca_cumulative_variance_report.png"):
    """
    Analyze the explained variance of PCA components to determine how many dimensions are needed to capture most of the variance.
    Args:
        features_matrix (numpy.ndarray): The matrix of extracted features (samples x features).
        destination_path (str): The path where the diagnostic plot will be saved.
    """
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


if __name__ == "__main__":
    # Load configuration
    config = load_config("config.yaml")
    
    # Extract paths from config, or use defaults
    dataset_cfg = config.get("dataset", {})
    train_dir = dataset_cfg.get("train_path", "./data/train")
    test_good_dir = dataset_cfg.get("test_good_path", "./data/test/good")
    test_defect_dir = dataset_cfg.get("test_reject_path", "./data/test/reject")

    # Set up the folder map with the EXACT labels needed for LDA masks
    folder_map = {
        train_dir: "Train - Good",
        test_good_dir: "Test - Good",
        test_defect_dir: "Test - Defect"
    }
    
    X, Y, paths = extract_features(folder_map, config) 
    
    if len(X) > 0:
        n_95=analyze_pca_variance(X, destination_path=config.get("paths", {}).get("pca_cumulative_variance_plot_path", "pca_cumulative_variance_report.png"))
        plot_interactive_tsne(X, Y, paths, n_components_pca=n_95, destination_path=config.get("paths", {}).get("interactive_tsne_plot_path", "interactive_tsne_plot.png"))
        realistic_pca_lda_analysis(X[Y == "Train - Good"], X[Y == "Test - Defect"], n_pca_components=n_95, destination_path=config.get("paths", {}).get("pca_lda_analysis_plot_path", "pca_lda_analysis.png"))
    else:
        print("No images found! Check the paths entered in your yaml or script.")