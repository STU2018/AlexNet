from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        with open(txt_path, 'r') as f:
            self.images = []
            self.transform = transform
            self.target_transform = target_transform

            for line in f:
                line = line.rstrip()  # strip '\n' in the end
                string = line.split(' ')
                self.images.append((string[0], int(string[1])))  # a tuple: (image path,label), label is int format

    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label  # (tensor, int)

    def __len__(self):
        return len(self.images)


def get_dataloader():
    train_data = MyDataset(
        txt_path='./data/catVSdog/train.txt',
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    test_data = MyDataset(
        txt_path='./data/catVSdog/test.txt',
        transform=transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader
