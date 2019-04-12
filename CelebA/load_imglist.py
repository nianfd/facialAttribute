import torch.utils.data as data

from PIL import Image, ImageEnhance
import cv2
import os
import os.path
import random
import numpy as np

def default_loader(path, target):
    # print(path)
    img = Image.open(path)
    #labels = np.array(target, dtype=np.float32)
    labels = np.array(target)
    return img, labels





class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, loader=default_loader):
        self.root      = root
        self.transform = transform
        self.loader    = loader
        with open(fileList, 'r') as file:
            self.lines = file.readlines()
        random.shuffle(self.lines)

    def __getitem__(self, index):
        curLine = self.lines[index]
        curLine = curLine[:-1]
        splitLineContents = curLine.split(' ')
        imgPath=splitLineContents[0]
        #print(imgPath)
        landmarks = splitLineContents[1:]
        labels = []
        for id in range(len(landmarks)):
            if landmarks[id] == '1':  #具有本属性，标签为0，否则为1
                labels.append(int(0))
            else:
                labels.append(int(1))
        #print(landmarks)
        #print(labels)
        img, target = self.loader(os.path.join(self.root, imgPath), labels)


        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.lines)