import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import os

from net import AlexNet
from val import val_runner
from train import train_runner
from dataloader import get_dataloader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    epochs = 30
    train_loader, val_loader = get_dataloader()

    best_val_acc = 0.0
    print('start to train model')
    for epoch in range(1, epochs + 1):
        train_runner(model, device, train_loader, optimizer, epoch)
        val_acc = val_runner(model, device, val_loader)
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            if os.path.exists('save') == False:
                os.mkdir('save')
            torch.save(model, './save/alexnet-model.pth')
    print('train done')
