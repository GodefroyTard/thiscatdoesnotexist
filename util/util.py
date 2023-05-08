import torchvision
import cv2
import time
import torch.nn
import os

def label_sampel(batch_size,n_class,device):
    label = torch.LongTensor(batch_size, 1).random_()%n_class
    one_hot= torch.zeros(batch_size, n_class).scatter_(1, label, 1)
    return label.squeeze(1).to(device), one_hot.to(device)     

def D_hinge(real,fake):
    d_loss_real = torch.nn.ReLU()(1.0 - real).mean()
    d_loss_fake = torch.nn.ReLU()(1.0 + fake).mean()
    return d_loss_real,d_loss_fake

def G_hinge(fake):
    m = torch.nn.ReLU()
    return m(- fake.mean())

def save_tensor_batch(batch,path=None,name=None):

    print('Shape :', batch.shape)
    grid = torchvision.utils.make_grid(batch)
    if name is None:
        name = str(time.time())+'.png'
    if path is None:
        path ='.img_debug/'
    torchvision.utils.save_image(grid,os.path.join(path,name))

def makegrid(batch):

    grid = torchvision.utils.make_grid(batch)
    return grid

    
