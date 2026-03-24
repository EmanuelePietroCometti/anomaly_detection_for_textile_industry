import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from config import load_config
from torchmetrics.classification import BinaryAUROC

class DenseBackboneWrapper(nn.Module):
    """
    Wrapper to remove Global Average Pooling and apply a 1x1 convolutional head
    for dense (spatial) classification.
    """
    def __init__(self, backbone_name):
        super(DenseBackboneWrapper, self).__init__()
        print(f"Loading backbone: {backbone_name}")
        
        try:
            model_func = getattr(models, backbone_name)
            self.backbone = model_func(weights='DEFAULT')
        except AttributeError:
            raise ValueError(f"Backbone '{backbone_name}' not supported by torchvision.models")

        # Automatically detect and disable Pooling/FC layers
        if hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                in_features = self.backbone.classifier[-1].in_features
            else:
                in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            if hasattr(self.backbone, 'avgpool'):
                self.backbone.avgpool = nn.Identity()
            self.arch_type = 'features'
            
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            if hasattr(self.backbone, 'avgpool'):
                self.backbone.avgpool = nn.Identity()
            self.arch_type = 'resnet'
        else:
            raise ValueError("Unknown backbone architecture.")

        # Spatial Projection Head (1x1 Convolutions)
        self.dense_head = nn.Sequential(
            nn.Conv2d(in_features, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1) # Output: 1 channel for anomaly probability
        )

    def forward(self, x):
        # Extract features keeping spatial topology [Batch, Channels, Height, Width]
        if self.arch_type == 'features':
            features = self.backbone.features(x)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            features = self.backbone.layer4(x)
            
        # Spatial logits [Batch, 1, Height, Width]
        logits_map = self.dense_head(features)
        return logits_map

def apply_selective_freezing(model):
    """
    Freeze early feature extractors and unfreeze mid/late layers used by PatchCore.
    """
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False 
        
        if model.arch_type == 'features' and "features" in name:
            try:
                block_idx = int(name.split('.')[1])
                # Unfreeze intermediate EfficientNet blocks (e.g., from 4 onwards)
                if block_idx >= 4:  
                    param.requires_grad = True
            except ValueError:
                pass
        elif model.arch_type == 'resnet' and "layer" in name:
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
                
    # Always train the dense head
    for param in model.dense_head.parameters():
        param.requires_grad = True

def train_custom_resnet(config_dict):
    tl_config = config_dict['transfer_learning']
    model_config = config_dict['model_architecture']
    
    data_dir = tl_config['data_dir']
    # Fallback to 'val' if val_data_dir is not explicitly set in config
    val_data_dir = tl_config.get('val_data_dir', data_dir.replace('train', 'val'))
    
    num_epochs = tl_config['num_epochs']
    batch_size = tl_config['batch_size']
    lr = tl_config['learning_rate']
    save_path = tl_config['save_path']
    backbone_name = model_config['backbone']

    # Train transforms with RandomCrop for local feature focus
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transforms: deterministic CenterCrop
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = datasets.ImageFolder(root=val_data_dir, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseBackboneWrapper(backbone_name).to(device)
    apply_selective_freezing(model)

    # BCE Loss to map probability on each spatial pixel
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = torch.amp.GradScaler(device.type)

    # Spatial AUROC Metric initialization
    spatial_auroc_metric = BinaryAUROC().to(device)

    print(f"Starting Dense Training on {device} for {num_epochs} epochs...")
    best_val_auroc = 0.0
    
    for epoch in range(num_epochs):
        # ==========================
        # TRAINING PHASE
        # ==========================
        model.train()
        train_running_loss = 0.0
        
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type):
                logits_map = model(images) 
                
                # Dynamic spatial label expansion
                B, _, H, W = logits_map.shape
                spatial_labels = labels.view(B, 1, 1, 1).expand(B, 1, H, W)
                
                loss = criterion(logits_map, spatial_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_running_loss += loss.item()
            
        train_loss = train_running_loss / len(train_dataloader)

        # ==========================
        # VALIDATION PHASE
        # ==========================
        model.eval()
        val_running_loss = 0.0
        spatial_auroc_metric.reset()
        
        with torch.no_grad():
            for val_images, val_labels in val_dataloader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                with torch.amp.autocast(device_type=device.type):
                    val_logits_map = model(val_images)
                    
                    B, _, H, W = val_logits_map.shape
                    val_spatial_labels = val_labels.view(B, 1, 1, 1).expand(B, 1, H, W)
                    
                    v_loss = criterion(val_logits_map, val_spatial_labels.float())
                    
                val_running_loss += v_loss.item()

                # Metric Update
                val_probs_map = torch.sigmoid(val_logits_map)
                preds_flat = val_probs_map.view(-1)
                labels_flat = val_spatial_labels.reshape(-1).long()
                
                spatial_auroc_metric.update(preds_flat, labels_flat)
                
        val_loss = val_running_loss / len(val_dataloader)
        epoch_auroc = spatial_auroc_metric.compute().item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Spatial AUROC: {epoch_auroc:.4f}")

        # Model Checkpointing based on AUROC
        if epoch_auroc > best_val_auroc:
            best_val_auroc = epoch_auroc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.backbone.state_dict(), save_path)
            print(f" -> New AUROC peak! Backbone weights saved to: {save_path}")

if __name__ == "__main__":
    config = load_config() 
    train_custom_resnet(config)