import torch
from torch.utils.data import Dataset
import cv2
import os  

class FacadesDataset(Dataset):
    def __init__(self, path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')  
    
        images = [f for f in os.listdir(path) if f.lower().endswith(image_extensions)]  
         
        self.image_filenames = [os.path.join(path, image) for image in images]  
        
    def __len__(self): 
        return len(self.image_filenames)
    
    def __getitem__(self, idx): 
        
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0
        image_rgb = image[:, :, :256]
        image_semantic = image[:, :, 256:]
        return image_rgb, image_semantic