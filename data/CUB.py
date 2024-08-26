import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
class CUB():
    def __init__(self, root, is_train=True, data_len=None,transform=None, target_transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))

        img_name_list = []
        for line in img_txt_file:

            img_name_list.append(line[:-1].split(' ')[-1])


        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))


        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        train_label_list = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        test_label_list = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        if self.is_train:
           
            self.train_img = [np.array(Image.open(os.path.join(self.root, 'images', train_file))) for train_file in
                              train_file_list[:data_len]]
            
            self.train_label = train_label_list
        if not self.is_train:
            self.test_img = [np.array(Image.open(os.path.join(self.root, 'images', test_file))) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list
    
    def __getitem__(self,index):
        
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
        
        else:
            img, target = self.test_img[index], self.test_label[index]

        if len(img.shape) == 2:
            
            img = np.stack([img]*3,2)
        
        img = Image.fromarray(img,mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)
def load_cub200(train_batch_size, test_batch_size):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])    



    train_set = CUB(root = "/data/dataset/CUB_200_2011/", is_train=True, transform=transform_train)
    test_set = CUB(root = "/data/dataset/CUB_200_2011/", is_train=False, transform=transform_test)
    print(f"train set len {len(train_set)}")
    print(f"test set len {len(test_set)}")
    # examine_count(train_label_count, name = "train")
    # examine_count(test_label_count, name = "test")

    kwargs = {'num_workers': 2, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_set ,
                    batch_size=train_batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(test_set,
                    batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader