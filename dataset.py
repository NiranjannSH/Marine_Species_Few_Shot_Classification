import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import IMAGE_SIZE, IMAGE_EXTS, SEED, EPISODE_DIR

class TwoCropTransform:
    def __init__(self, base):
        self.base = base
    def __call__(self, x):
        return [self.base(x), self.base(x)]

train_transform = TwoCropTransform(
    transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.9),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
        transforms.RandomApply([transforms.RandomRotation(15)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
)

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SupConDataset(Dataset):
    def __init__(self, root, train_classes, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(train_classes)}

        for cls in train_classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith(IMAGE_EXTS):
                    self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def class_split(config, seed=SEED):
    """Sorted class folders shuffled with random.Random(seed) -> (base, validation, novel) classes."""
    root = config["root"]
    classes = sorted(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)))
    random.Random(seed).shuffle(classes)
    train_cnt, val_cnt = config["train_classes_cnt"], config["val_classes_cnt"]
    return classes[:train_cnt], classes[train_cnt:train_cnt + val_cnt], classes[train_cnt + val_cnt:]

def episode_path(dataset_name, split, shot):
    return os.path.join(EPISODE_DIR, f"{dataset_name}_{split}_{shot}shot_episodes.pt")

def load_episodes(path):
    """Fixed episodes written by make_episodes.py -> list of (support, query, labels)."""
    episodes = torch.load(path, map_location="cpu", weights_only=False)
    return [(e["support"].float(), e["query"].float(), e["labels"].long()) for e in episodes]
