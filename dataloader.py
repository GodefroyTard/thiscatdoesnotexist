
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import os
import cv2
import numpy as np


def showimg(img,kp=None,box=None):
    if kp is not None:
        for point in kp:
            img = cv2.circle(img, point, radius=5, color=(0, 0, 255), thickness=-1)
    if box is not None:
        for b in box:
            img = cv2.circle(img, b, radius=5, color=(0, 255, 0), thickness=-1)
    
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

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
                if img.endswith(".jpg"):
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
        keypointspath = imgpath+'.cat'
        try:
            f = open(keypointspath, "r")
            img = cv2.imread(imgpath)
            keypoints = [int(kp) for kp in  f.read().split()]
        except:
            print('Erreur ouverture !')
        kp1= (keypoints[5],keypoints[6]) # bouche
        kp2= (keypoints[9],keypoints[10]) #oreille gauche
        kp3= (keypoints[15],keypoints[16]) #oreille droite

        kp4= (keypoints[1],keypoints[2]) #oeil gauche
        kp5= (keypoints[3],keypoints[4]) #oeil droit

        vec1 = (kp5[0]-kp4[0],kp5[1]-kp4[1])
        vec2 = (vec1[0],0)

        angle = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        if np.isnan(angle):
            angle=0


        x1 = (min(kp1[0],kp2[0],kp3[0]),min(kp1[1],kp2[1],kp3[1]))
        x2 = (max(kp1[0],kp2[0],kp3[0]),max(kp1[1],kp2[1],kp3[1]))

        dist = np.sqrt((kp2[0]-kp3[0])**2 + (kp2[1]-kp3[1])**2)

        x1_ = (max(int(x1[0]-0.10*dist),0),max(int(x1[1]-0.10*dist),0) )
        x2_ = (min(int(x2[0]+0.10*dist),img.shape[1]),min(int(x2[1]+0.10*dist),img.shape[0]) )

        # if x1 != x1_ or x2 != x2_:
        #     print('diff')

        try:
            crop =img[x1_[1]:x2_[1],x1_[0]:x2_[0]]

            crop = rotate_image(crop,int(angle*180/3.1415))


            img = cv2.resize(crop,(128,128))
            img = self.transforms(img)
            #img = img/255.0
        except:
            print('erreur image : ' + imgpath)
            

        sample = {'image': img }

        return sample