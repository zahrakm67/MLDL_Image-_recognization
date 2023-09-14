import matplotlib.pyplot as plt
import torch
import numpy as np
def plot_pic(dataloader,labelmap,r = False,is_continous =False ):
  fig, axs = plt.subplots(4,4,figsize=(15,15))
  
  image,depth,label = next(iter(dataloader))

  if is_continous:
      
      for i in range(0,16,2):
        img = image[i].squeeze()
        dpt = depth[i].squeeze()
        rotation_angle =  label[i]

        plt.subplot(4,4,i+1)
        plt.tight_layout()
        
        plt.imshow(img.permute(1, 2, 0) , cmap="gray",interpolation='none')
        
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4,4,i+2)
        plt.tight_layout()
        
        plt.imshow(dpt.permute(1, 2, 0) , cmap="gray",interpolation='none')
        if r :
              plt.title("depth rotation is {} degree ".format(rotation_angle ))
        else :
              plt.title(" {} depth ".format(rotation_angle))
        plt.xticks([])
        plt.yticks([])

      return(fig)
  else :


    
    for i in range(0,16,2):
      img = image[i].squeeze()
      dpt = depth[i].squeeze()
      rotation_angle =  label[i]


      label_numpy = label[i].numpy()
      index = list(np. where(label_numpy == 1))
      

    
      try:
            lbl = [key for key in labelmap if (labelmap[key] == index[0])]
      except:
          lbl=['bell_pepper'] 

      
      plt.subplot(4,4,i+1)
      plt.tight_layout()
      
      plt.imshow(img.permute(1, 2, 0) , cmap="gray",interpolation='none')
      if r:
        pass

      else:
            plt.title("{}".format(lbl[0]))
      plt.xticks([])
      plt.yticks([])

      plt.subplot(4,4,i+2)
      plt.tight_layout()
      
      plt.imshow(dpt.permute(1, 2, 0) , cmap="gray",interpolation='none')
      if r :
            plt.title("depth rotation is: {} ".format(lbl[0]))
      else :
            plt.title(" {} depth ".format(lbl[0]))
      plt.xticks([])
      plt.yticks([])

    return(fig)
