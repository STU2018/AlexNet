import torch
import torch.optim as optim
import os

from net import AlexNet
from test import test_runner
from train import train_runner
from dataloader import get_dataloader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    train_loader, test_loader = get_dataloader()

    best_train_acc = 0.0
    print('start to train model')
    for epoch in range(1, epochs + 1):
        train_acc = train_runner(model, device, train_loader, optimizer, epoch)
        if best_train_acc < train_acc:
            best_train_acc = train_acc
            if os.path.exists('save') == False:
                os.mkdir('save')
            torch.save(model, './model/alexnet-model.pth')
    print('train done')

    print('start to test model')
    test_runner(model, device, test_loader)
    print('test done')
