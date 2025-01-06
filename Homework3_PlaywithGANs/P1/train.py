import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import *
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
   
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(gen,disc, dataloader, optimizer_d,optimizer_g, device, epoch, num_epochs):
    for i, (image_rgb,image_semantic) in enumerate(dataloader):

        image_rgb = image_rgb.to(device) 

        image_semantic = image_semantic.to(device) 
        noise=torch.randn_like(image_semantic).to(device)
        real_label=torch.ones(image_semantic.shape[0],1).to(device)
        fake_label=torch.zeros(image_semantic.shape[0],1).to(device)

        fake=gen(noise,image_semantic).detach()
        real_out = disc(image_rgb,image_semantic)
        fake_out = disc(fake,image_semantic) 
        optimizer_d.zero_grad()
        loss_real_D = nn.BCELoss()(real_out, real_label)
        loss_fake_D = nn.BCELoss()(fake_out, fake_label)               
        loss_D=loss_fake_D+loss_real_D
        loss_D.backward()
        optimizer_d.step()
        


        noise=torch.randn_like(image_semantic).to(device)
        gen_imgs = gen(noise,image_semantic)                                             
        out = disc(gen_imgs,image_semantic)                                    
        optimizer_g.zero_grad()
        loss_G = nn.BCELoss()(out, real_label)+0.25*nn.MSELoss()(gen_imgs,image_rgb)
        loss_G.backward()
        optimizer_g.step()

        
        if epoch % 200 == 0 and i == 0:
            save_images(image_semantic, image_rgb, gen_imgs, 'train_results', epoch)
       
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss_g: {loss_G.item():.6f}, Loss_d: {loss_D.item():.6f}')


        

def validate(gen,disc, dataloader,  device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """

    val_loss_d = 0.0
    val_loss_g = 0.0
    with torch.no_grad():
        for i, (image_rgb,image_semantic) in enumerate(dataloader):

            image_rgb = image_rgb.to(device) 
            image_semantic = image_semantic.to(device)
            noise=torch.randn_like(image_semantic).to(device)
            real_label=torch.ones([image_semantic.shape[0],1],dtype=torch.float32).to(device)
            fake_label=torch.zeros([image_semantic.shape[0],1],dtype=torch.float32).to(device)

            fake=gen(noise,image_semantic).detach()
            real_out = disc(image_rgb,image_semantic)
            fake_out = disc(fake,image_semantic)

            loss_real_D = nn.BCELoss()(real_out, real_label)
            loss_fake_D = nn.BCELoss()(fake_out, fake_label)              
            loss_D=loss_fake_D+loss_real_D
            val_loss_d+=loss_D.item()

            noise=torch.randn_like(image_semantic).to(device)
            gen_imgs = gen(noise,image_semantic)                                            
            out = disc(gen_imgs,image_semantic)                                    
            loss_G = nn.BCELoss()(out, real_label)
            val_loss_g+=loss_G.item()

    

            if epoch % 200 == 0 and i == 0:
                save_images(image_semantic, image_rgb, gen_imgs, 'val_results', epoch)



    
    avg_val_loss_g = val_loss_g / len(dataloader)
    avg_val_loss_d = val_loss_d / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Gen Validation Loss: {avg_val_loss_g:.6f}, Disc Validation Loss: {avg_val_loss_d:.6f}')



def main():
    """
    Main function to set up the training and validation processes.
    """
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    
    train_dataset = FacadesDataset('./cityscapes/train')
    val_dataset = FacadesDataset('./cityscapes/val')

    train_loader = DataLoader(train_dataset, batch_size=360, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=360, shuffle=False, num_workers=4)

   
    gen = cGAN_gen(3,3).to(device)
    disc = cGAN_disc(3).to(device)
 
    optimizer_g = optim.Adam(gen.parameters(), lr=0.0005,betas=(0.5, 0.999))
    optimizer_d = optim.Adam(disc.parameters(), lr=0.0005,betas=(0.5, 0.999))

    
    scheduler1 = StepLR(optimizer_g, step_size=100, gamma=0.6)
    scheduler2 = StepLR(optimizer_d, step_size=100, gamma=0.6)

   
    num_epochs = 1601
    for epoch in range(num_epochs):
        train_one_epoch(gen,disc, train_loader, optimizer_d,optimizer_g, device, epoch, num_epochs)
        validate(gen,disc, val_loader,  device, epoch, num_epochs)

        scheduler1.step()
        scheduler2.step()

        if (epoch + 1) % 400 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(gen.state_dict(), f'checkpoints/gen_epoch_{epoch + 1}.pth')
            torch.save(disc.state_dict(), f'checkpoints/disc_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
