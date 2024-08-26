import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
def load_pets37(train_batch_size, test_batch_size):
    transform_train = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224, padding =16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])    

    pets_train_dataset = datasets.ImageFolder(root='/data/datasets/pets/trainval/', transform=transform_train)
    pets_test_dataset = datasets.ImageFolder(root='/data/datasets/pets/test/', transform=transform_test)
    kwargs = {'num_workers': 2, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(pets_train_dataset ,
                    batch_size=train_batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(pets_test_dataset,
                    batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader