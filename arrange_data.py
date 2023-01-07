import os


def generate_txt(txt_path, image_path):
    with open(txt_path, 'w') as f:
        sub_dirs = os.listdir(image_path)

        for sub_dir in sub_dirs:
            image_file_list = os.listdir(os.path.join(image_path, sub_dir))

            for index in range(len(image_file_list)):
                label = image_file_list[index].split('.')[0]

                if label == 'cat':
                    label = '0'
                else:
                    label = '1'

                this_line = os.path.join(image_path, sub_dir, image_file_list[index]) + ' ' + label + '\n'
                f.write(this_line)


def generate_dataset_txt():
    train_txt_path = os.path.join("data", "catVSdog", "train.txt")
    train_image_path = os.path.join("data", "catVSdog", "train_data")
    valid_txt_path = os.path.join("data", "catVSdog", "test.txt")
    valid_image_path = os.path.join("data", "catVSdog", "test_data")
    generate_txt(train_txt_path, train_image_path)
    generate_txt(valid_txt_path, valid_image_path)


if __name__ == '__main__':
    generate_dataset_txt()
