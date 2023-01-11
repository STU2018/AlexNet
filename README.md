# AlexNet
Pytorch implementation of AlexNet.



# Results

The trained AlexNet model reaches about **85.6%** accuracy rate on verification set, and can reach a nice accuracy rate on the images I collect by myself.



# Requirements

```
torch==1.6.10
torchvision==0.7.0
```



# File management

├─data                                                                      \
│  ├─catVSdog--------folder to store images of train and val dataset   \
│  │  ├─test_data                                                           \
│  │  │  ├─cat                                                              \
│  │  │  └─dog                                                              \
│  │  └─train_data                                                          \
│  │      ├─cat                                                             \
│  │      └─dog                                                             \
│  └─pictures----------folder to store images of test dataset            \
└─save-----------------folder to save model and test results             



# Usage

### Prepare data

The first thing to note is that the training and validation sets should come from large data sets that are publicly available and of a certain size, and the test sets can be images that you want to classify.

Put your images in folder `./data/catVSdog `. Note that the cat and dog images should be placed in their respective folders and divided into training set and verification set. **Note that:** each of your dog image should be named like _dog.xxx.format_ (start with word **dog**, xxx can be any string, format represent the format of the image), same as cat image.

Put your test images in folder `./data/picutres `.Don't worry about the name of the file, because an automated script handles it for you.

### Run

clone this repository

```
git clone git@github.com:STU2018/AlexNet.git
```

 create conda environment

```
conda create -n AlexNet python==3.7
conda activate AlexNet
```

`cd` to the root directory of the repository，and install requirements

```
cd AlexNet
pip install -r requirement.txt
```

generate txt label file

```
python arrange_data.py
```

train the AlexNet model

```
python run.py
```

test on your own images

```
python test_on_own_pictures.py
```

# Reference

[Zhihu 李大熊] : 手撕 CNN 之 AlexNet (PyTorch 实战篇) [link](https://zhuanlan.zhihu.com/p/495615011) 



