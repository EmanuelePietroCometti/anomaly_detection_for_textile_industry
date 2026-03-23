import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import os
from config import load_config

def process_five_crops(crops):
    """
    Module-level function to process crops. 
    Required because lambda functions cannot be pickled by Windows multiprocessing.
    """
    crop_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return torch.stack([crop_transform(crop) for crop in crops])

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class ProjectionHead(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=2048, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.head(x)    
        return F.normalize(x, dim=1)

def train_custom_resnet(config_dict):
    """
    Function that trains a custom backbone using parameters from config.yaml.
    """
    tl_config = config_dict['transfer_learning']
    model_config = config_dict['model_architecture']
    
    data_dir = tl_config['data_dir']
    num_epochs = tl_config['num_epochs']
    batch_size = tl_config['batch_size']
    lr = tl_config['learning_rate']
    temperature = tl_config['temperature']
    out_dim = tl_config['out_dim']
    save_path = tl_config['save_path']
    backbone_name = model_config['backbone']

    transform = transforms.Compose([
        transforms.FiveCrop(256),
        transforms.Lambda(process_five_crops)
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print(f"Loading backbone: {backbone_name}")
    try:
        # Dynamic loading using recent torchvision APIs (weights='DEFAULT')
        model_func = getattr(models, backbone_name)
        model = model_func(weights='DEFAULT')
    except AttributeError:
        raise ValueError(f"Backbone '{backbone_name}' not supported by torchvision.models")

    # Dynamic adaptation of the classification head based on the architecture
    if hasattr(model, 'fc'):
        # ResNet-style architectures
        in_features = model.fc.in_features
        model.fc = ProjectionHead(in_features=in_features, out_dim=out_dim)
        
        for name, param in model.named_parameters():
            if name.startswith('conv1') or name.startswith('bn1') or name.startswith('layer1'):
                param.requires_grad = False
                
    elif hasattr(model, 'classifier'):
        # EfficientNet / MobileNet-style architectures
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = ProjectionHead(in_features=in_features, out_dim=out_dim)
        else:
            in_features = model.classifier.in_features
            model.classifier = ProjectionHead(in_features=in_features, out_dim=out_dim)
            
        for name, param in model.named_parameters():
            if name.startswith('features.0') or name.startswith('features.1') or name.startswith('features.2'):
                param.requires_grad = False
    else:
        print("Warning: Unknown model architecture. No head modification applied.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = SupConLoss(temperature=temperature)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = torch.amp.GradScaler(device.type)

    print(f"Starting training on {device} for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            bs, n_views, c, h, w = images.size()
            images_flat = images.view(-1, c, h, w).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type):
                features_flat = model(images_flat) 
                features_3d = features_flat.view(bs, n_views, -1)
                loss = criterion(features_3d, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(dataloader):.4f}")
    
    # Restore the final layer to Identity to save only the weights useful for PatchCore
    if hasattr(model, 'fc'):
        model.fc = nn.Identity()
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Identity()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Custom weights successfully saved: {save_path}")

if __name__ == "__main__":
    config = load_config() 
    train_custom_resnet(config)