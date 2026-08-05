import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        return self.encoder(x)

class SupConModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Encoder()
        
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projector(feat)
        return F.normalize(proj, dim=1)

class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return x @ w.t()

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = features.float()
        labels = labels.float()
        device = features.device
        _, num_views, _ = features.shape

        features = F.normalize(features, dim=2)
        feats = torch.cat(torch.unbind(features, dim=1), dim=0)

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(num_views, num_views)

        sim = torch.matmul(feats, feats.T) / self.temperature
        logits_mask = torch.ones_like(sim) - torch.eye(sim.shape[0], device=device)

        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_stable = sim - sim_max.detach()

        exp_sim = torch.exp(sim_stable) * logits_mask
        pos_mask = mask * logits_mask

        log_prob = sim_stable - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-7)

        pos_count = pos_mask.sum(dim=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-7)

        valid_anchors = (pos_count > 0).float()
        return - (mean_log_prob_pos * valid_anchors).sum() / (valid_anchors.sum() + 1e-7)