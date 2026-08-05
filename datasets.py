import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import IMAGE_SIZE, IMAGE_EXTS, SEED

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

class FewShotDataset:
    def __init__(self, config, split_seed=SEED, sampler_seed=SEED):
        self.config = config
        root = config["root"]

        classes = sorted([name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))])
        
        split_rng = random.Random(split_seed)
        split_rng.shuffle(classes)
        
        self.rng = random.Random(sampler_seed)

        train_cnt = config["train_classes_cnt"]
        val_cnt = config["val_classes_cnt"]

        self.train_classes = classes[:train_cnt]
        self.val_classes = classes[train_cnt: train_cnt + val_cnt]
        self.test_classes = classes[train_cnt + val_cnt:]

        self.data = {}
        for class_name in classes:
            class_dir = os.path.join(root, class_name)
            img_paths = sorted(os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTS))

            self.data[class_name] = []
            for path in img_paths:
                try:
                    with Image.open(path) as img:
                        tensor_img = eval_transform(img.convert("RGB"))
                    self.data[class_name].append(tensor_img)
                except Exception:
                    continue

    def _sample_images(self, images, count):
        if len(images) >= count:
            return self.rng.sample(images, count)
        return self.rng.choices(images, k=count)

    def sample_episode(self, split, shot, query_per_class):
        pool = self.test_classes if split == "test" else self.val_classes
        way = self.config["test_way"] if split == "test" else self.config["val_way"]

        selected_classes = self.rng.sample(pool, way)
        support, query, labels = [], [], []

        for label, class_name in enumerate(selected_classes):
            chosen = self._sample_images(self.data[class_name], shot + query_per_class)
            for img_tensor in chosen[:shot]:
                support.append(img_tensor)
            for img_tensor in chosen[shot:]:
                query.append(img_tensor)
                labels.append(label)

        return torch.stack(support), torch.stack(query), torch.tensor(labels, dtype=torch.long)