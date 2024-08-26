from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs 
            self.transform = transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open('/data/datasets/oxford-102-flowers/'+fn).convert('RGB')


        if self.transform is not None:
            img = self.transform(img)

        
        return img, label
    def __len__(self):
        return len(self.imgs)

def load_flowers102(train_batch_size, test_batch_size):
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224, padding =16),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))
    ])
    test_transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
    ])
    train_path="/data/datasets/oxford-102-flowers/train.txt"
    train_set=MyDataset(txt_path=train_path,transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True, num_workers=2
    )

    test_path = "/data/datasets/oxford-102-flowers/valid.txt"
    test_set = MyDataset(txt_path=test_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader