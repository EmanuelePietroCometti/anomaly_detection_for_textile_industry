import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from config import load_config
from torchmetrics.classification import BinaryAUROC
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize, RandomCrop, CenterCrop
from torchvision.transforms import InterpolationMode
import logging
from lightning.pytorch import seed_everything

logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)
seed_everything(42, workers=True)

class StandardBackboneWrapper(nn.Module):
    """
    Standard classifier wrapper for Transfer Learning.
    Learns domain-specific features using Global Average Pooling (GAP).
    """
    def __init__(self, backbone_name):
        super(StandardBackboneWrapper, self).__init__()
        print(f"Loading backbone: {backbone_name}")
        
        try:
            model_func = getattr(models, backbone_name)
            self.backbone = model_func(weights='DEFAULT')
        except AttributeError:
            raise ValueError(f"Backbone '{backbone_name}' not supported")

        # Replace the final classification head and track architecture type
        if hasattr(self.backbone, 'classifier'): # EfficientNet
            if isinstance(self.backbone.classifier, nn.Sequential):
                in_features = self.backbone.classifier[-1].in_features
            else:
                in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, 1)
            self.arch_type = 'features'
            self.head_name = 'classifier'
            
        elif hasattr(self.backbone, 'fc'): # ResNet
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 1)
            self.arch_type = 'resnet'
            self.head_name = 'fc'
        else:
            raise ValueError("Unknown backbone architecture.")

    def forward(self, x):
        # Forward pass returns a single logit per image [Batch, 1]
        return self.backbone(x)

def apply_selective_freezing(model):
    """
    Freeze early feature extractors and unfreeze mid/late layers and the classification head.
    """
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False 
        
        # Unfreeze intermediate/late layers
        if model.arch_type == 'features' and "features" in name:
            try:
                block_idx = int(name.split('.')[1])
                if block_idx >= 4:  
                    param.requires_grad = True
            except ValueError:
                pass
        elif model.arch_type == 'resnet' and "layer" in name:
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
                
        # Always train the newly initialized classification head
        if model.head_name in name:
            param.requires_grad = True

def train_custom_resnet(config_dict):
    tl_config = config_dict['transfer_learning']
    model_config = config_dict['model_architecture']
    
    data_dir = tl_config['data_dir']
    val_data_dir = tl_config.get('val_data_dir', data_dir.replace('train', 'val'))
    
    num_epochs = tl_config['num_epochs']
    batch_size = tl_config['batch_size']
    lr = tl_config['learning_rate']
    save_path = tl_config['save_path']
    backbone_name = model_config['backbone']

    gen_config = config_dict.get("general_configuration", {})
    image_size = gen_config.get("image_size", (512, 512))
    crop_size = gen_config.get("crop_size", (380, 380))

    if "efficientnet" in backbone_name:
        train_transform = transforms.Compose([
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = Compose([
            Resize(image_size),
            RandomCrop(crop_size),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = Compose([
            Resize(image_size),
            CenterCrop(crop_size),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = datasets.ImageFolder(root=val_data_dir, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandardBackboneWrapper(backbone_name).to(device)
    apply_selective_freezing(model)

    # Standard BCE Loss for 1D logits
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = torch.amp.GradScaler(device.type)

    # Standard Image-level AUROC Metric
    auroc_metric = BinaryAUROC().to(device)

    print(f"Starting Training on {device} for {num_epochs} epochs...")
    best_val_auroc = 0.0
    
    for epoch in range(num_epochs):
        # ==========================
        # TRAINING PHASE
        # ==========================
        model.train()
        train_running_loss = 0.0
        
        for images, labels in train_dataloader:
            images = images.to(device)
            # Reshape labels to match [Batch, 1] output of the model
            labels = labels.to(device).float().view(-1, 1) 

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                logits = model(images) 
            loss = criterion(logits.float(), labels)
            
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
        auroc_metric.reset()
        
        with torch.no_grad():
            for val_images, val_labels in val_dataloader:
                val_images = val_images.to(device)
                # Ensure labels match model output
                val_labels = val_labels.to(device).float().view(-1, 1)

                val_logits = model(val_images)
                v_loss = criterion(val_logits.float(), val_labels)
                    
                val_running_loss += v_loss.item()

                # Metric Update: compute probabilities and pass to metric
                val_probs = torch.sigmoid(val_logits)
                auroc_metric.update(val_probs, val_labels.long())
                
        val_loss = val_running_loss / len(val_dataloader)
        epoch_auroc = auroc_metric.compute().item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {epoch_auroc:.4f}")

        # Model Checkpointing based on AUROC
        if epoch_auroc > best_val_auroc:
            best_val_auroc = epoch_auroc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save the entire backbone state dict (which Patchcore will easily load)
            torch.save(model.backbone.state_dict(), save_path)
            print(f" -> New AUROC peak! Backbone weights saved to: {save_path}")

if __name__ == "__main__":
    config = load_config() 
    train_custom_resnet(config)