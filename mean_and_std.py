import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    channels_sum_d, channels_squared_sum_d = 0, 0
    for data,depth, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

        channels_sum_d += torch.mean(depth, dim=[0,2,3])
        channels_squared_sum_d += torch.mean(depth**2, dim=[0,2,3])
        

        

    
    mean = channels_sum / num_batches
    mean_d =  channels_sum_d / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    std_d = ( channels_squared_sum_d / num_batches - mean_d ** 2) ** 0.5
    return mean, std,mean_d,std_d