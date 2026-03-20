import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import os


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
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
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

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
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

def train_custom_resnet(data_dir, num_epochs=10, batch_size=16):
    """
    Function that train a custom resnet with a transfer learning approach.
    Parameters:
        - data_dir: directory where there is the dataset for the training
        - num_classes: in the trasfer lerning approach for AD the NN must be trained as for a binary classification where the two classes are good and reject. The number of classes for AD must be setted to 2
        - num_epochs: number of epochs for which iterate the training loop. In transfer learning approach, it is not necessary to use a high value of training epochs
        - batch_size: number of images that are processed in parallel
    """
    # Transformation to adapt the input images to the input that is expected by the resnet (224x224)
    # To preserve the image resolution a five crop approach is used (4 corners and 1 center crop)
    transform = transforms.Compose([
        transforms.FiveCrop(256),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(crop) for crop in crops
        ]))
    ])

    dataset = datasets.ImageFolder(root=data_dir,transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = ProjectionHead(in_features=in_features, out_dim=128)

    for name, param in model.named_parameters():
        if name.startswith('conv1') or name.startswith('bn1') or name.startswith('layer1'):
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = SupConLoss(temperature=0.07)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler(device.type)

    print(f"Starting training on {device} for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            # FiveCrop input: [batch_size, 5, 3, 224, 224]
            bs, n_views, c, h, w = images.size()
            
            # Input flattening
            # Wide_Resnet50_2 accept only 4 dimensions: [batch_size * 5, 3, 224, 224]
            images_flat = images.view(-1, c, h, w).to(device)
            labels = labels.to(device) # NON replicare più le etichette!

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
    model.fc = nn.Identity()

    os.makedirs("custom_weights", exist_ok=True)
    save_path = "custom_weights/resnet_finetuned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Custom weights sucessfully saved: {save_path}")

if __name__ == "__main__":
    train_custom_resnet(data_dir="./dataset_classificazione", num_classes=3, num_epochs=20, batch_size=16)