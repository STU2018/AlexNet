from PIL import Image
import os
import torch
from torchvision import transforms


def deal_test_imgs():
    path = './data/pictures'
    file_list = os.listdir(path)

    file_index = 0
    for file_name in file_list:
        new_name = '.'.join([str(file_index).zfill(2), 'png'])
        os.rename(os.path.join(path, file_name), os.path.join(path, new_name))
        file_index = file_index + 1


if __name__ == '__main__':
    deal_test_imgs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load('./save/alexnet-model.pth')
    model = model.to(device)
    model.eval()

    classes = ('cat', 'dog')

    images_path = './data/pictures'
    images_list = os.listdir(images_path)

    with open('./save/pre_results.txt', 'w') as f:
        for image_name in images_list:
            this_image_path = os.path.join(images_path, image_name)
            img = Image.open(this_image_path).convert('RGB')

            image_transform = transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            img = image_transform(img)
            img = img.to(device)
            img = img.unsqueeze(0)  # [channel, H, W] -> [batch_size, channel, H, W]

            output = model(img)

            predict = output.argmax(dim=1)
            value, predicted = torch.max(output.data, 1)
            pred_class = classes[predicted.item()]

            f.write(image_name + ' ' + pred_class + '\n')
