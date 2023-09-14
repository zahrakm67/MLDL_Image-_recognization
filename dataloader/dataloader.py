# import skimage.io as io
import os
# import pandas as pd
import random
# torch 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.models as models






class CustomImageDataset(Dataset):
    def __init__(self,dt, annotations_list,label_map, img_dir,depth_dir, transform=None,transform_d=None, target_transform=None):
        self.dt = dt
        self.img_labels =  annotations_list
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.transform_d = transform_d
        self.target_transform = target_transform
        self.label_map =label_map

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        if self.dt == 'ROD':
          img_path = os.path.join(self.img_dir, self.img_labels[i].split('***')[0]+'crop.png')
          depth_path = os.path.join(self.depth_dir, self.img_labels[i].split('***')[0]+'depthcrop.png')
          image = Image.open(img_path)           
          depth = Image.open(depth_path)        
          pre_label = self.img_labels[i].split('/')[0]
          l =self.label_map[pre_label]
        elif self.dt == 'SYNROD':          
          img_path =   os.path.join(self.img_dir, (self.img_labels[i].replace('***','rgb')).split(' ')[0])
          depth_path = os.path.join(self.depth_dir, (self.img_labels[i].replace('***','depth')).split(' ')[0])
          image = Image.open(img_path)           
          depth = Image.open(depth_path)  
          pre_label = self.img_labels[i].split('/')[0]
          if pre_label == 'bell_papper':
            pre_label = 'bell_pepper'

          l =self.label_map[pre_label]
        label = torch.zeros([len(self.label_map)])
        label[l] = 1

        if self.transform:
            image = self.transform(image)
        if self.transform_d:
            depth = self.transform_d(depth)
        if self.target_transform:
            label = self.target_transform(label)
        return image,depth, label

#.........................................
class roteted_dataset(Dataset):
    def __init__(self,dt, annotations_list, img_dir,depth_dir, transform=None,transform_d=None):
        self.dt = dt
        self.img_labels =  annotations_list
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.transform_d = transform_d
        
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        if self.dt == 'ROD':
          img_path = os.path.join(self.img_dir, self.img_labels[i].split('***')[0]+'crop.png')
          depth_path = os.path.join(self.depth_dir, self.img_labels[i].split('***')[0]+'depthcrop.png')
          image = Image.open(img_path)           
          depth = Image.open(depth_path)  

          k = random.randint(0, 3)  
          j =  random.randint(0, 3)  

          image = image.rotate(90*k, expand=True)
          depth = depth.rotate(90*j, expand=True)

          rotation = j-k
          if rotation == -1:
            rotation = 3
          elif  rotation == -2:
             rotation = 2
          elif  rotation == -3:
            rotation = 1

          label_R = torch.zeros(4)
          label_R[rotation] = 1
          


        elif self.dt == 'SYNROD':          
          img_path =   os.path.join(self.img_dir, (self.img_labels[i].replace('***','rgb')).split(' ')[0])
          depth_path = os.path.join(self.depth_dir, (self.img_labels[i].replace('***','depth')).split(' ')[0])
          image = Image.open(img_path)           
          depth = Image.open(depth_path) 

          k = random.randint(0, 3)  
          j=   random.randint(0, 3)  

          image = image.rotate(90*k, expand=True)
          depth = depth.rotate(90*j, expand=True)

          rotation = j-k
          if rotation == -1:
            rotation = 3
          elif  rotation == -2:
             rotation = 2
          elif  rotation == -3:
            rotation = 1
          
          
          label_R = torch.zeros(4)
          label_R[rotation] = 1
          

        if self.transform:
            image = self.transform(image)
        if self.transform_d:
            depth = self.transform_d(depth)
      
        
        return image,depth, label_R
  



#.........................................


class continous_rotation_dataset(Dataset):
    def __init__(self,dt, annotations_list, img_dir,depth_dir, transform=None,transform_d=None):
        self.dt = dt
        self.img_labels =  annotations_list
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.transform_d = transform_d
        
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        if self.dt == 'ROD':
          img_path = os.path.join(self.img_dir, self.img_labels[i].split('***')[0]+'crop.png')
          depth_path = os.path.join(self.depth_dir, self.img_labels[i].split('***')[0]+'depthcrop.png')
          image = Image.open(img_path)           
          depth = Image.open(depth_path)  

          k = random.randint(0, 360)  
          j =  random.randint(0, 360)  

          image = image.rotate(k, expand=True)
          depth = depth.rotate(j, expand=True)

          rotation = j-k
          if rotation < 0:
            rotation = rotation +360
     

          


        elif self.dt == 'SYNROD':          
          img_path =   os.path.join(self.img_dir, (self.img_labels[i].replace('***','rgb')).split(' ')[0])
          depth_path = os.path.join(self.depth_dir, (self.img_labels[i].replace('***','depth')).split(' ')[0])
          image = Image.open(img_path)           
          depth = Image.open(depth_path) 

          # k = random.randint(0, 3)  
          # j=   random.randint(0, 3)  

          # image = image.rotate(90*k, expand=True)
          # depth = depth.rotate(90*j, expand=True)

          k = random.randint(0, 360)  
          j =  random.randint(0, 360)  

          image = image.rotate(k, expand=True)
          depth = depth.rotate(j, expand=True)

          rotation = j-k
          if rotation < 0:
            rotation = rotation +360
          

        if self.transform:
            image = self.transform(image)
        if self.transform_d:
            depth = self.transform_d(depth)
      
        
        return image,depth, rotation 


  