# MLDL_Image_recognization

This repository contains the code for our adaptation of the paper [**Unsupervised Domain Adaptation through Inter-modal Rotation for RGB-D Object Recognition**](https://arxiv.org/abs/2004.10016)

The goal of the project is to induce Domain Adaptation from label-rich to label-poor distributions via an auxiliary self-supervised task that integrates RGB image data with the corresponding Depth maps. 

We start by studying and re-implementing the relative rotation method described in [this paper](https://arxiv.org/abs/2004.10016), along with executing hyperparameter optimization and comparisons with a **source-only** (no adaptation) baseline. 

As a next step, we proposed a variation for the project from a set of possible ideas.

The original paper proposes two different DA settings:
- **synROD ➞ ROD**
- **synHB ➞ valHB**

For the scope of this project, you are required to run experiments for the **first setting only**.

# DataSet
The two datasets used are the object classification datasets **SynROD** and **ROD**. SynROD will be the source domain, for which you have the labels, and ROD will be the target domain, for which you can’t use the labels during the training phase and use the labels for evaluation.
You can access the datasets through this [**ROD and SynROD**](https://www.dropbox.com/s/xdy5cfu7m63pk46/ROD-synROD.tar?dl=0).

Since image recognition networks only accept three-channel images (RGB), and a depth map only has one channel, you are provided colorized depth maps which you can use as depth images for the network. The depth maps have been colorized using the surface normals technique. To download the data you can use the following command:

> wget 'https://www.dropbox.com/s/xdy5cfu7m63pk46/?dl=1' -O \ROD-synROD.tar

The datasets are distributed as tar files, you will have to load and untar them in Colab.

- **Training sets** = SynROD (labeled), ROD (without object labels)
- **Validation set** = **Test set** = ROD (with object labels)

# Implementation
1. Implement the synROD ➞ ROD source-only baseline in PyTorch.
     a. only e2e experiments described in the paper
2. Implement the synROD ➞ ROD Domain Adaptation procedure described in the paper.
3. Run appropriate hyperparameter optimization for both source-only and Domain Adaptation. For each method you have to choose at least 5 different sets of hyperparameters.
4. Choose a variation of the project by implementing a different DA(Domain Adaptation) method that we call it **Implementation of Different Rotation Based Methods for Unsupervised Domain Adaptation**
<div align="center">
  <img src="https://github.com/zahrakm67/MLDL_Image_Recognization/blob/main/imgs/RGB-D DA.jpg" alt="Alt text" title="The general structure of our method for RGB-D DA" width="600" height="200">
  </div>
  
<caption>
Here we can observe the general structure of our method for RGB-D DA. Firstly, the blue squares represent the CNN, composed by a two-stream feature extractor E and by two network heads, the main and the pretext, respectively M and P. The first is trained using object recognition with the labeled source data (see the red arrow), while the second exploits both source and target samples. Finally, both RGB and Depth images are independently rotate before being transferred to the network.
</caption>

   
# Dependencies
1. Python 3 (tested on python 3.6)
2. PyTorch
3. With GPU TensorFlow

# Additional useful resources
- Implementations of popular self-supervised algorithms available at:
  
   https://github.com/jason718/awesome-self-supervised-learning
- Implementations of popular domain adaptation algorithms available at:
  
   https://github.com/zhaoxin94/awesome-domain-adaptation
- Additional material for RGB-D integration:
  
   http://ais.informatik.uni-freiburg.de/publications/papers/eitel15iros.pdf
- A survey on Domain Adaptation algorithms:
  
   https://arxiv.org/pdf/1802.03601.pdf
