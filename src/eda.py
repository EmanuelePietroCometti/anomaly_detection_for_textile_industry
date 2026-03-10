import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer,f1_score, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_validate
import seaborn as sns
from config import load_config
from sklearn.preprocessing import StandardScaler



def extract_features(folder_map, config):
    # Initialize ResNet18 (Feature Extractor only)
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
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

def plot_interactive_tsne(features, labels, image_paths):
    print("Dimensionality reduction with PCA + t-SNE...")
    
    pca = PCA(n_components=min(50, len(features)))
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
    plt.show()

def test_lda_separability(features, labels):
    print("\n--- LINEAR SEPARABILITY TEST (LDA) ---")
    
    # Filter only "Good" and "Defect" classes for LDA
    mask_good = (labels == "Train - Good") | (labels == "Test - Good")
    mask_defects = (labels == "Test - Defect")
    
    if not any(mask_good) or not any(mask_defects):
        print("Error: Missing classes. Make sure 'Good' and 'Defect' data is loaded.")
        return 0.0

    X_binary = np.vstack((features[mask_good], features[mask_defects]))
    Y_binary = np.concatenate((np.zeros(sum(mask_good)), np.ones(sum(mask_defects))))
    
    # Train the LDA (Finds the best separating hyperplane)
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_binary, Y_binary)
    
    # Calculate the theoretical accuracy of this surface
    score = lda.score(X_binary, Y_binary)
    print(f"Accuracy of the found linear surface: {score * 100:.2f}%")
    
    return score

def lda_cross_validation(features, labels):
    print("\n--- LINEAR SEPARABILITY TEST (LDA) - TRUTH DETECTOR ---")
    
    mask_good = (labels == "Train - Good") | (labels == "Test - Good")
    mask_defects = (labels == "Test - Defect")
    
    if not any(mask_good) or not any(mask_defects):
        print("Error: Missing classes.")
        return

    X_binary = np.vstack((features[mask_good], features[mask_defects]))
    Y_binary = np.concatenate((np.zeros(sum(mask_good)), np.ones(sum(mask_defects))))
    
    print(f"Total samples: {len(Y_binary)} (Good: {sum(mask_good)}, Defects: {sum(mask_defects)})")
    print(f"Number of dimensions (features): {X_binary.shape[1]}")
    
    # Retrieve the "fake" accuracy of the linear surface on the same data (overfitting check)
    score_overfit = test_lda_separability(features, labels)
    print(f"\n[WARNING] 'Fake' Accuracy (Train and Test on same data): {score_overfit * 100:.2f}%")
    
    # PCA: Reduce to 20 dimensions to prevent overfitting and make the problem more realistic for cross-validation
    n_components = 240
    print(f"\nApplying PCA: Reducing dimensions from {X_binary.shape[1]} to {n_components} for Cross-Validation...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_binary)
    
    # Cross-validation with LDA as the classifier
    lda = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'balanced_acc': make_scorer(balanced_accuracy_score),
        'f1': make_scorer(f1_score)
    }
    
    # Perform cross-validation and compute metrics
    scores = cross_validate(lda, X_pca, Y_binary, cv=skf, scoring=scoring)
    
    bal_acc_mean = scores['test_balanced_acc'].mean()
    f1_mean = scores['test_f1'].mean()
    
    print(f"[REAL] Cross-Validation Balanced Accuracy: {bal_acc_mean * 100:.2f}% (Baseline: 50%)")
    print(f"[REAL] Cross-Validation F1-Score: {f1_mean:.4f} (Perfect: 1.0000)")
    
    # Analyze results and provide diagnosis
    if f1_mean < 0.5 and score_overfit > 0.95:
        print("\n❌ DIAGNOSIS: SEVERE OVERFITTING. The 100% was an illusion caused by too many dimensions.")
        print("-> Conclusion: The defects are NOT linearly separable on unseen data. You MUST use Anomaly Detection (EfficientAD).")
    elif f1_mean >= 0.85:
        print("\n✅ DIAGNOSIS: REAL SEPARATION. The defects are genuinely linearly separable even with strict metrics.")
        print("-> Conclusion: Check for trivial artifacts (lighting/cropping). If none exist, your problem is extremely easy.")
    else:
        print("\n⚠️ DIAGNOSIS: MODERATE SEPARATION. The linear model struggles on unseen data.")
        print("-> Conclusion: EfficientAD remains the most robust choice to handle your variations.")

def analyze_pca_variance(features_matrix):
    """
    Analyze the explained variance of PCA components to determine how many dimensions are needed to capture most of the variance.
    Args:
        features_matrix (numpy.ndarray): The matrix of extracted features (samples x features).
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
        plot_interactive_tsne(X, Y, paths)
        lda_cross_validation(X, Y)
    else:
        print("No images found! Check the paths entered in your yaml or script.")