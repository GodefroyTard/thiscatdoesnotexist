
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import os
import cv2

class CatDataset(Dataset):
    """cat dataset."""
    def __init__(self,N,root_dir):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        path = []

        for n in N :
            folder = os.path.join(root_dir,n)
            L = os.listdir(folder)

            for img in L :
                path.append( os.path.join(folder,img) )

        #os.listdir()
        self.path = path
        self.root_dir = root_dir


        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.dataaug = None

        print("dataset created ! ")

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):

        imgpath = self.path[idx]
        try:
            img = cv2.imread(imgpath)
            img = cv2.resize(img,(128,128))
            img = self.transforms(img)
            #img = img/255.0
        except:
            print('erreur image : ' + imgpath)

        sample = {'image': img }

        return sample