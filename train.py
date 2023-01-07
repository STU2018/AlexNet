import torch.nn.functional as F


def train_runner(model, device, train_loader, optimizer, epoch):
    model.train()

    total, correct, train_loss, batch_num = 0, 0, 0.0, 0

    for batch_index, (inputs, labels) in enumerate(train_loader):
        batch_num = batch_num + 1
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = F.cross_entropy(outputs, labels)
        train_loss += loss.item()
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Train Epoch {} done".format(epoch))

