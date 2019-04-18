#!/usr/bin/env python
# coding: utf-8


# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# 
# Training Torchvision Models for Star Wars
# =============================
# 
# based upon pytorch tutorials
# 
# **Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__
# 
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# 
# 
# 

# In this tutorial we will take a deeper look at how to finetune and
# feature extract the `torchvision
# models <https://pytorch.org/docs/stable/torchvision/models.html>`__, all
# of which have been pretrained on the 1000-class Imagenet dataset. This
# tutorial will give an indepth look at how to work with several modern
# CNN architectures, and will build an intuition for finetuning any
# PyTorch model. Since each model architecture is different, there is no
# boilerplate finetuning code that will work in all scenarios. Rather, the
# researcher must look at the existing architecture and make custom
# adjustments for each model.
# 
# In this document we will perform two types of transfer learning:
# finetuning and feature extraction. In **finetuning**, we start with a
# pretrained model and update *all* of the model’s parameters for our new
# task, in essence retraining the whole model. In **feature extraction**,
# we start with a pretrained model and only update the final layer weights
# from which we derive predictions. It is called feature extraction
# because we use the pretrained CNN as a fixed feature-extractor, and only
# change the output layer. For more technical information about transfer
# learning see `here <https://cs231n.github.io/transfer-learning/>`__ and
# `here <https://ruder.io/transfer-learning/>`__.
# 
# In general both transfer learning methods follow the same few steps:
# 
# -  Initialize the pretrained model
# -  Reshape the final layer(s) to have the same number of outputs as the
#    number of classes in the new dataset
# -  Define for the optimization algorithm which parameters we want to
#    update during training
# -  Run the training step
# 
# 
# 

# In[2]:


from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, sys
import copy
from datetime import datetime, date
import cv2
import paiv_utils.paiv_utils as pu
import pdb
import glob
import re
import subprocess

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import logging
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))

def runcmd(mycmd) :
    cmdary = re.split("\s+", mycmd)
    nprint(cmdary)
    process = subprocess.Popen(cmdary, stdout=subprocess.PIPE)
    stdout, err = process.communicate()
    # print(stdout)
    return stdout


# Data augmentation and normalization for training
# Just normalization for validation
def get_transforms(input_size) :
    data_transforms = {
        'train': transforms.Compose([
            #transforms.Resize(input_size),
            #transforms.RandomResizedCrop(input_size),
            transforms.RandomResizedCrop(input_size, scale=(1.0, 1.0),ratio=(1.0, 1.0)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Resize(input_size),
            #transforms.CenterCrop(input_size),
            transforms.RandomResizedCrop(input_size, scale=(1.0, 1.0),ratio=(1.0, 1.0)),        
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


# Inputs
# ------
# 
# Here are all of the parameters to change for the run. We will use the
# *hymenoptera_data* dataset which can be downloaded
# `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`__.
# This dataset contains two classes, **bees** and **ants**, and is
# structured such that we can use the
# `ImageFolder <https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder>`__
# dataset, rather than writing our own custom dataset. Download the data
# and set the ``data_dir`` input to the root directory of the dataset. The
# ``model_name`` input is the name of the model you wish to use and must
# be selected from this list:
# 
# ::
# 
#    [resnet, alexnet, vgg, squeezenet, densenet, inception]
# 
# The other inputs are as follows: ``num_classes`` is the number of
# classes in the dataset, ``batch_size`` is the batch size used for
# training and may be adjusted according to the capability of your
# machine, ``num_epochs`` is the number of training epochs we want to run,
# and ``feature_extract`` is a boolean that defines if we are finetuning
# or feature extracting. If ``feature_extract = False``, the model is
# finetuned and all model parameters are updated. If
# ``feature_extract = True``, only the last layer parameters are updated,
# the others remain fixed.
# 
# 
# 





# Helper Functions
# ----------------
# 
# Before we write the code for adjusting the models, lets define a few
# helper functions.
# 
# Model Training and Validation Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The ``train_model`` function handles the training and validation of a
# given model. As input, it takes a PyTorch model, a dictionary of
# dataloaders, a loss function, an optimizer, a specified number of epochs
# to train and validate for, and a boolean flag for when the model is an
# Inception model. The *is_inception* flag is used to accomodate the
# *Inception v3* model, as that architecture uses an auxiliary output and
# the overall model loss respects both the auxiliary output and the final
# output, as described
# `here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__.
# The function trains for the specified number of epochs and after each
# epoch runs a full validation step. It also keeps track of the best
# performing model (in terms of validation accuracy), and at the end of
# training returns the best performing model. After each epoch, the
# training and validation accuracies are printed.
# 
# 
# 

# In[5]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    from time import gmtime, strftime
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        t =  strftime("%Y-%m-%d %H:%M:%S", gmtime() ) 
        print('{}  Epoch {}/{}'.format(t, epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('{}  Epoch {}/{}'.format(t, epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            temp = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
            logging.info(temp)  
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
        logging.info(" ")        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    temp = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    logging.info(temp)
    temp = 'Best val Acc: {:4f}'.format(best_acc)
    temp = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    logging.info(temp)    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
# 
# 
# 

# In[6]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Initialize and Reshape the Networks
# -----------------------------------
# 
# Now to the most interesting part. Here is where we handle the reshaping
# of each network. Note, this is not an automatic procedure and is unique
# to each model. Recall, the final layer of a CNN model, which is often
# times an FC layer, has the same number of nodes as the number of output
# classes in the dataset. Since all of the models have been pretrained on
# Imagenet, they all have output layers of size 1000, one node for each
# class. The goal here is to reshape the last layer to have the same
# number of inputs as before, AND to have the same number of outputs as
# the number of classes in the dataset. In the following sections we will
# discuss how to alter the architecture of each model individually. But
# first, there is one important detail regarding the difference between
# finetuning and feature-extraction.
# 
# When feature extracting, we only want to update the parameters of the
# last layer, or in other words, we only want to update the parameters for
# the layer(s) we are reshaping. Therefore, we do not need to compute the
# gradients of the parameters that we are not changing, so for efficiency
# we set the .requires_grad attribute to False. This is important because
# by default, this attribute is set to True. Then, when we initialize the
# new layer and by default the new parameters have ``.requires_grad=True``
# so only the new layer’s parameters will be updated. When we are
# finetuning we can leave all of the .required_grad’s set to the default
# of True.
# 
# Finally, notice that inception_v3 requires the input size to be
# (299,299), whereas all of the other models expect (224,224).
# 
# Resnet
# ~~~~~~
# 
# Resnet was introduced in the paper `Deep Residual Learning for Image
# Recognition <https://arxiv.org/abs/1512.03385>`__. There are several
# variants of different sizes, including Resnet18, Resnet34, Resnet50,
# Resnet101, and Resnet152, all of which are available from torchvision
# models. Here we use Resnet18, as our dataset is small and only has two
# classes. When we print the model, we see that the last layer is a fully
# connected layer as shown below:
# 
# ::
# 
#    (fc): Linear(in_features=512, out_features=1000, bias=True) 
# 
# Thus, we must reinitialize ``model.fc`` to be a Linear layer with 512
# input features and 2 output features with:
# 
# ::
# 
#    model.fc = nn.Linear(512, num_classes)
# 
# Alexnet
# ~~~~~~~
# 
# Alexnet was introduced in the paper `ImageNet Classification with Deep
# Convolutional Neural
# Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`__
# and was the first very successful CNN on the ImageNet dataset. When we
# print the model architecture, we see the model output comes from the 6th
# layer of the classifier
# 
# ::
# 
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     ) 
# 
# To use the model with our dataset we reinitialize this layer as
# 
# ::
# 
#    model.classifier[6] = nn.Linear(4096,num_classes)
# 
# VGG
# ~~~
# 
# VGG was introduced in the paper `Very Deep Convolutional Networks for
# Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`__.
# Torchvision offers eight versions of VGG with various lengths and some
# that have batch normalizations layers. Here we use VGG-11 with batch
# normalization. The output layer is similar to Alexnet, i.e.
# 
# ::
# 
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     )
# 
# Therefore, we use the same technique to modify the output layer
# 
# ::
# 
#    model.classifier[6] = nn.Linear(4096,num_classes)
# 
# Squeezenet
# ~~~~~~~~~~
# 
# The Squeeznet architecture is described in the paper `SqueezeNet:
# AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
# size <https://arxiv.org/abs/1602.07360>`__ and uses a different output
# structure than any of the other models shown here. Torchvision has two
# versions of Squeezenet, we use version 1.0. The output comes from a 1x1
# convolutional layer which is the 1st layer of the classifier:
# 
# ::
# 
#    (classifier): Sequential(
#        (0): Dropout(p=0.5)
#        (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
#        (2): ReLU(inplace)
#        (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
#     ) 
# 
# To modify the network, we reinitialize the Conv2d layer to have an
# output feature map of depth 2 as
# 
# ::
# 
#    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
# 
# Densenet
# ~~~~~~~~
# 
# Densenet was introduced in the paper `Densely Connected Convolutional
# Networks <https://arxiv.org/abs/1608.06993>`__. Torchvision has four
# variants of Densenet but here we only use Densenet-121. The output layer
# is a linear layer with 1024 input features:
# 
# ::
# 
#    (classifier): Linear(in_features=1024, out_features=1000, bias=True) 
# 
# To reshape the network, we reinitialize the classifier’s linear layer as
# 
# ::
# 
#    model.classifier = nn.Linear(1024, num_classes)
# 
# Inception v3
# ~~~~~~~~~~~~
# 
# Finally, Inception v3 was first described in `Rethinking the Inception
# Architecture for Computer
# Vision <https://arxiv.org/pdf/1512.00567v1.pdf>`__. This network is
# unique because it has two output layers when training. The second output
# is known as an auxiliary output and is contained in the AuxLogits part
# of the network. The primary output is a linear layer at the end of the
# network. Note, when testing we only consider the primary output. The
# auxiliary output and primary output of the loaded model are printed as:
# 
# ::
# 
#    (AuxLogits): InceptionAux(
#        ...
#        (fc): Linear(in_features=768, out_features=1000, bias=True)
#     )
#     ...
#    (fc): Linear(in_features=2048, out_features=1000, bias=True)
# 
# To finetune this model we must reshape both layers. This is accomplished
# with the following
# 
# ::
# 
#    model.AuxLogits.fc = nn.Linear(768, num_classes)
#    model.fc = nn.Linear(2048, num_classes)
# 
# Notice, many of the models have similar output structures, but each must
# be handled slightly differently. Also, check out the printed model
# architecture of the reshaped network and make sure the number of output
# features is the same as the number of classes in the dataset.
# 
# 
# 

# In[7]:


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size







############################################################################################################
# Inference Funcs
############################################################################################################
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

    # # EXAMPLE USAGE:
    # # instantiate the dataset and dataloader
    # data_dir = "your/data_dir/here"
    # dataset = ImageFolderWithPaths(data_dir) # our custom dataset
    # dataloader = torch.utils.DataLoader(dataset)
    # 
    # # iterate over data
    # for inputs, labels, paths in dataloader:
    #     # use the above variables freely
    #     print(inputs, labels, paths)



# This will return a dictionary of filenames -> className output
def infer_spectrograms(model_path, spectrogram_dir, batch_size=1) : 

    nprint("Passed Parameters :")
    nprint("model_path      : {}".format(model_path))
    nprint("spectrogram_dir : {}".format(spectrogram_dir))
    #nprint("class_map       : {}".format(class_map))
    nprint("batch_size      : {}".format(batch_size))



    #data_dir = "./tmp/"
    #data_dir = "./data/bananas_data"
    
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    #model_name = "squeezenet"
    model_name = "inception"
    
    # Load previously saved model data , not yet applied to model arch ...
    saved_model = torch.load(model_path)
    for i in saved_model.keys() :
        nprint("Model keys = {}".format(i))

    # Number of classes in the dataset
    class_map = saved_model['classmap']
    inv_class_map = {v: k for k, v in class_map.items()}
    num_classes = len(inv_class_map)

    nprint("Inverting class map for predictions to get mapped to class name")
    nprint("inv_class_map       : {}".format(inv_class_map))
    nprint("num_classes      : {}".format(num_classes))



    

    
    # Flag for feature extracting. When False, we finetune the whole model, 
    #   when True we only update the reshaped layer params

    # copy pngs to infer/unknown directory
    # pdb.set_trace()
    os.system("mkdir -p {}".format(spectrogram_dir+"infer/unknown/"))
    pnglist = glob.glob(spectrogram_dir+"*.png")
    for i in pnglist :
        os.system('cp {} {}'.format(i,spectrogram_dir+"infer/unknown/"))

    feature_extract = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the model for this run
    inf_model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    inf_model_ft.to(device)
    
    # Get the data transformer used from validation.  DONT USE TRAINING HERE
    data_transforms = get_transforms(input_size)['val']

    # Print the model we just instantiated
    nprint("Base model ({}) loaded (random parameters).  Num state_dict tensors = {}".format(model_name,len(list(inf_model_ft.state_dict()))))
    
    # Set to Inference mode
    nprint("Now loading saved checkpoint parameters from {} {} ".format(os.getcwd(), model_path))

    inf_model_ft.eval()


    # Load state_dict
    inf_model_ft.load_state_dict(saved_model['model_state_dict'])
    
    # Create Inference dataset
    nprint("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    # under tmp is infer/unknown/*pngs 

    infer_dataset2 = ImageFolderWithPaths(spectrogram_dir, data_transforms)


    # Create training and validation dataloaders
    infer_dataloader2 = torch.utils.data.DataLoader(infer_dataset2, batch_size=batch_size, shuffle=True, num_workers=4)

    # Run inference and build dictionary
    rv_list = []
    for inputs, labels, paths in infer_dataloader2:
        inputs = inputs.to(device)
        outputs = inf_model_ft(inputs)
        # Grab the best label
        # pdb.set_trace()
        probs = torch.softmax(outputs,1)
        winner = torch.argmax(probs,1)
        
        #print(outputs)
        winner_np = winner.cpu().detach().numpy()
        class_pred = [inv_class_map[x] for x in winner_np]
        probs_np = probs.cpu().detach().numpy()
        probs_dict = [ {inv_class_map[0] : x[0], inv_class_map[1] : x[1],inv_class_map[2] : x[2]} for x in probs_np]
        # Grab just the filenames ...
        fns = [x.split('/')[-1] for x in paths]

        # For each batch, just glue on the current list to overall filename -> pred 
        # this is not pretty, 0 -> class_pred, 1-> confidence_ary
        prediction_conf_tuple = zip(class_pred, probs_dict)
        rv_list += list(zip(fns, prediction_conf_tuple))

    nprint("Ran inference on {} files.  Here are the results".format(len(rv_list)))
    print(rv_list)
    rv_dict = dict(rv_list)
    return rv_dict




############################################################################################################
# Video Funcs
############################################################################################################
    
#annotate_video(input_video , args.splitsize, annotations, output_file)
def get_fn(filename, frame_number, fps, total_secs, split_size=1, extension="png") :
    [fn_base,junk] = filename.split('.')
    final_number = int(total_secs)
    
    idx_number = int((float(frame_number) / float(fps * total_secs)) * int(total_secs)) + 1 #indexed from 1

    #nprint("{}{}{}".format(frame_number ,fps ,total_secs))
    #idx_number = idx_number / split_size
    #nprint("{}".format(idx_number))

    rv_string = "{}-{}-of-{}.{}".format(fn_base, idx_number,final_number,extension)
    return rv_string




def annotate_video(input_video, split_size, annotations_dict, output_directory, output_fn, sample_rate =1, max_frames=10000):

    #Error checking
    if(not(os.path.isfile(input_video))) :
        nprint("Error : Input File {} does not exist.  Check path".format(input_video))
        return 1;

    if(re.search("mp4$", input_video) == None) :
        nprint("Error : Input File {} must be of type mp4 (need to add support for other formats)".format(input_video))
        return 1;


    print(input_video)


    loopcnt = 1 # loopcnt set to one since we read the first frame
    # Second Pass over video
    spectrogram_dir = output_directory + "/infer/unknown/"
    nprint("Annotating {} and saving in {}".format(input_video, output_directory))
    cap  = cv2.VideoCapture(input_video)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) # fps = video.get(cv2.CAP_PROP_FPS)
    secs = total_frames / fps
    nprint("Total number of frames  = {} (frames)".format(total_frames))
    nprint("Frame rate              = {} (fps)".format(fps))
    nprint("Total seconds for video = {} (s)".format(secs))

    if(max_frames > total_frames) :
        max_frames = total_frames
        print("Processing number of frames  = {} (frames)".format(max_frames))
    ret, frame = cap.read()

    output  = cv2.VideoWriter(output_directory + "/" + output_fn, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1],frame.shape[0]), True)

    # Used to properly index into json list
    # Label use annotations ... 
    sample_rate_idx = 0
    
    while(loopcnt < max_frames ):
        ret, frame = cap.read()
        
        # Frame striding .....
        if(loopcnt % sample_rate == 0 and frame is not None) :
            # plot_image( frame )
            nprint("{} {} {}".format(loopcnt ,fps ,secs))

            file_name_idx = get_fn(input_video, loopcnt, fps, secs, split_size=1, extension="png") 
            (classification , confidence_dict) = annotations_dict[file_name_idx]
            #nprint("confidence_dict = {}".format(confidence_dict))
            print_str = []
            print_str.append("Source PNG     : {}".format(file_name_idx))
            print_str.append("Classification : {}".format(classification))
            print_str.append("Confidence :")
            for k in sorted(confidence_dict.keys()) :
                print_str.append("  {0:<15s} : {1:.3f}".format(k, confidence_dict[k]))
            #nprint(print_str)

            frame = pu.draw_text_box(frame, "Inference Results : ", print_str ) 
            img_thumbnail = spectrogram_dir + file_name_idx
            nprint("img_thumbnail = {}".format(img_thumbnail))
            frame = pu.add_image_thumbnail(frame, img_thumbnail ) 
            output.write(frame)
            sample_rate_idx += 1

        loopcnt += 1

        if(loopcnt % sample_rate == 0 ) :
            nprint("Complete {} frames".format(loopcnt))
    cap.release()
    output.release()

    fn= output_directory + "/" + output_fn
    if(sample_rate == 1) :
        nprint("Adding audio track back into annotated movie ")
        nprint("1. Grab audio from original track and write to aac file")
        runcmd("ffmpeg -loglevel panic -y -i {} {}".format(input_video,input_video.replace("mp4", "aac")))
        fn_orig= fn
        fn= output_directory + "/a_v_" + output_fn
        nprint("2. Adding track onto annotated video")
        runcmd("ffmpeg -loglevel panic -y -i {} -i {} -map 0:v -map 1:a -c copy -shortest {}".format(fn_orig,input_video.replace("mp4", "aac"),fn))
    


    nprint("Program Complete : Wrote new movie : {}".format(fn))


