import torch
import torch.nn.functional as F


def val_runner(model, device, test_loader):
    model.eval()

    correct, test_loss, total = 0.0, 0.0, 0
    with torch.no_grad():
        # fixed bug: when not use enumerate, val-acc would maintain 0.5 or grow slowly
        for batch_index, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            test_loss += F.cross_entropy(outputs, labels).item()
            predict = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

    print("val-acc: {:.2f}%".format(100.0 * (correct / total)))
    return correct / total
