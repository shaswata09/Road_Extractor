
import os
from os.path import join
import datetime
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import torch.nn.functional as F

from imageLoader import RoadDatasetFolder, default_loader

import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imshow(img):
    #imshow(outputs[0])
    #print(torch.round(outputs[0]))
    with torch.no_grad():
        img = img.cpu()
        img = img /2 + 0.5 #unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.show()

def imgShowDouble(img1,img2):
    with torch.no_grad():
        img1 = img1.cpu().numpy()
        img2 = img2.cpu().numpy()

        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.imshow(np.rot90(img1,2))
        f.add_subplot(1,2,2)
        plt.imshow(np.rot90(img2,2))
        plt.show(block=True)


"""
UCNF Model

Input: N x N x 3 RGB image (512x512x3)

3x3 kernel size used across all convolution layers with a stride of 1x1 and padding of type same...
 - should be same spatial size (i.e. 512x512) after every convolution operation

 - followed by element wise activation ReLu

"""

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()

        self.conv1 = nn.Conv2d(3,16,3,padding="same")
        self.conv2 = nn.Conv2d(16,16,3,padding="same")
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(16,32,3,padding="same")
        self.conv4 = nn.Conv2d(32,64,3,padding="same")
        self.conv5 = nn.Conv2d(64,128,3,padding="same")
        self.conv6 = nn.Conv2d(192,64,3,padding="same")
        self.conv7 = nn.Conv2d(96,32,3,padding="same")
        self.conv8 = nn.Conv2d(48,16,3,padding="same")
        self.conv9 = nn.Conv2d(16,16,3,padding="same")
        self.finalConv = nn.Conv2d(16,1,1)
        #Deconvolutions
        self.deConv1 = nn.ConvTranspose2d(128,128,2,2)
        self.deConv2 = nn.ConvTranspose2d(64,64,2,2)
        self.deConv3 = nn.ConvTranspose2d(32,32,2,2)
    
    def forward(self,x):
    
        #[NumChannels x Height x Width]
        #Input: 3x512x512
        x = F.relu(self.conv1(x))
        #Output: 16x512x512

        skipCon3 = F.relu(self.conv2(x))
        #Output: 16x512x512

        x = self.pool(skipCon3)
        #Output: 16x256x256

        skipCon2 = F.relu(self.conv3(x))
        #Output: 32x256x256

        x = self.pool(skipCon2)
        #Output: 32x128x128


        skipCon1 = F.relu(self.conv4(x))
        #Output: [64x64x64]
        x = self.pool(skipCon1)
        
        #Output: [128x64x64]
        x = F.relu(self.conv5(x))

        #Transpose convlution, output: [128,128,128]
        x = self.deConv1(x)
        
        #Concatenate with first skipCon
        #Output: 192x128x128
        skipCon1 = torch.cat((x,skipCon1),1)

        #Output: 64x128x128
        x = F.relu(self.conv6(skipCon1))

        x = self.deConv2(x)
        #Output: 64x256x256

        #Skip con concatenation 2:
        # 64x256x256 + 32x256x256
        x = torch.cat((x,skipCon2),1)
        #Ouput: 96x256x256

        x = F.relu(self.conv7(x))
        #Output: 32x256x256

        x = self.deConv3(x)
        #Output: 32x512x512

        #Skip Con concatentation 3:
        # 32x512x512 + 16x512x512
        x = torch.cat((x,skipCon3),1)
        #Output: 48x512x512

        x = F.relu(self.conv8(x))
        #Output: 16x512x512

        x = F.relu(self.conv9(x))
        #Output: 16x512x512

        x = torch.sigmoid(self.finalConv(x))



        return x


def train_net(batch_size=3,save_path="./cnn.pth"):

    #Upon loading images we resize, convert to tensor, and normalize all the images
    sampleTransformations = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ])

    targetTransformations = transforms.Compose([
        transforms.Resize(512),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()#,
        #transforms.Normalize(mean=(0.5),std=(0.5))
    ])


    trainingDataSet = RoadDatasetFolder(os.getcwd() + "\\train",default_loader,transform=sampleTransformations,target_transform=targetTransformations)
    trainingDataLoader = data.DataLoader(dataset=trainingDataSet,batch_size=batch_size,shuffle=True)

    num_epochs = 750
    batch_size = 1
    learning_rate = .001

    model = ConvNet().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    sampleImages,targetImages = next(iter(trainingDataLoader))

    startTime = datetime.datetime.now()

    #n_total_steps = len(trainingDataLoader)
    for epoch in range(num_epochs):
        for i, (sampImages,targImages) in enumerate(trainingDataLoader):
            if i == 100:
                break

            sampleImages = sampImages.to(device)
            targetImages = targImages.to(device)

            #Forward pass
            outputs = model(sampleImages)

            loss = criterion(outputs,targetImages)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch[{epoch+1}], Step [{i+1}], Loss: {loss.item():.4f}')

            if epoch % 10 == 0:
                print(f"Curr time elapsed {datetime.datetime.now()-startTime}")

    print("Finished Training")


    PATH = save_path
    torch.save(model.state_dict(),PATH)



def test_net(load_path="cnn.pth"):

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(load_path))
    model.eval()

    #Upon loading images we resize, convert to tensor, and normalize all the images
    sampleTransformations = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ])

    targetTransformations = transforms.Compose([
        transforms.Resize(512),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()#,
        #transforms.Normalize(mean=(0.5),std=(0.5))
    ])


    testingDataSet = RoadDatasetFolder(os.getcwd() + "\\modules\\image_processing\\UFCN\\train",default_loader,transform=sampleTransformations,target_transform=targetTransformations)
    testingDataLoader = data.DataLoader(dataset=testingDataSet,batch_size=1,shuffle=True)

    for i, (sampImages,targImages) in enumerate(testingDataLoader):
        if i == 10:
            exit()
        
        sampleImages = sampImages.to(device)
        targetImages = targImages.to(device)

        #Forward pass
        outputs = model(sampleImages)
        #imshow(torch.round(outputs[0]))
        imgShowDouble(torch.round(outputs[0][0]),targetImages[0][0])


#Uncomment to run train_net
#train_net()

#test_net(load_path="cnn_100imgs_750epochs.pth")
test_net(load_path=join(os.getcwd(),"modules\\image_processing\\UFCN\\trainedModels\\cnn.pth"))
