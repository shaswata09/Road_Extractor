from torchvision.datasets import VisionDataset
import os
from os import listdir
from os.path import isfile, join
import torchvision.transforms as transforms

from torch.utils import data

from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class RoadDatasetFolder(VisionDataset):

    """
    A Datasetfolder class to load in all of the Road training and target images
    
    """

    def __init__(self,root,loader,extensions=None,transform=None,
                target_transform=None,is_valid_file=None):
        
        super(RoadDatasetFolder,self).__init__(root,transform=transform,
                                            target_transform=target_transform)
        


        self.loader = loader
        self.extensions = extensions

        #Each sample and target item is the filepath to that given image
        self.samples = []
        self.targets = []

        self._build_dataset(root)



    def _build_dataset(self,dir):
        """
        t
        """
        print("Building the dataset...")

        #Get all of the files found in the given input folder and split into either sample or target folder based off file name
        sampleFiles = []
        targetFiles = []
        for file in listdir(dir):
            if isfile(join(dir,file)):
                if "sat" in file:
                    sampleFiles.append(file)
                if "mask" in file:
                    targetFiles.append(file)
        #Sort incase the files got mixed up in folder
        sampleFiles.sort()
        targetFiles.sort()

        #Grab each of the sample and target files and ensure that their picture id matches up
        while len(sampleFiles) > 0 and len(targetFiles) > 0:
            sampleFile = sampleFiles.pop(0)
            targFile = targetFiles.pop(0)
            
            #This is not very robust :/
            #If the pic ID matches up for the sample and target then add to the samples and target lists
            if sampleFile.split("_")[0] == targFile.split("_")[0]:
                self.samples.append(join(dir,sampleFile))
                self.targets.append(join(dir,targFile))

    
        print("Finished building the dataset!")
        return
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self,index: int): #Return a Tuple[Any,Any]

        sample = self.loader(self.samples[index])
        target = self.loader(self.targets[index])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)


        return sample, target








