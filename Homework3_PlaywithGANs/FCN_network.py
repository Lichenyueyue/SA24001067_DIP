import torch
import torch.nn as nn
from einops import rearrange

def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=1): 
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.LeakyReLU()
    )
    return convlayer

def conv2d_bn_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=1): 
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.LeakyReLU()
    )
    return convlayer

def deconv_tanh(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.Tanh()
    )
    return convlayer



    
class rebuild_encoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(rebuild_encoder,self).__init__()
        self.channels=in_channels
        self.conv_stack0 = torch.nn.Sequential(
            conv2d_bn_relu(self.channels*2,16,6,2,2), 
            conv2d_bn_relu(16,16,3)
        )
        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(16,64,6,2,2),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(64,256,6,2,2), 
            conv2d_bn_relu(256,256,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(256,512,6,2,2), 
            conv2d_bn_relu(512,512,3)
        )

        self.downres=conv2d_bn_relu(self.channels*2,512,24,16,4)
    def forward(self, data,label):
        x=torch.cat([data,label],dim=1)
        conv1_out = self.conv_stack0(x)
        conv2_out = self.conv_stack1(conv1_out)
        conv3_out = self.conv_stack2(conv2_out)
        conv4_out = self.conv_stack3(conv3_out)+self.downres(x)
        return x,conv1_out,conv2_out,conv3_out,conv4_out
    
class rebuild_decoder(torch.nn.Module):
    def __init__(self, out_channels):
        super(rebuild_decoder,self).__init__()
        self.channels=out_channels
        self.deconv_4 = deconv_relu(512,800,3,1,1)  
        self.deconv_3 = deconv_relu(800,800,6,2,2) 
        self.deconv_2 = deconv_relu(256+800,320,6,2,2) 
        self.deconv_1 = deconv_relu(320+64,90,6,2,2)
        self.deconv_0 = deconv_relu(90+16,44,6,2,2) 
        self.out=nn.Sequential(
            deconv_relu(44+6,24,3,1,1),
            deconv_relu(24,8,3,1,1),
            deconv_sigmoid(8,self.channels,1,1,0),
        )

    def forward(self, x,conv1_out,conv2_out,conv3_out,conv4_out): 
        de4=self.deconv_4(conv4_out)
        de3=self.deconv_3(de4)
        de2=self.deconv_2(torch.cat([de3,conv3_out],dim=1))
        de1=self.deconv_1(torch.cat([de2,conv2_out],dim=1))
        de0=self.deconv_0(torch.cat([de1,conv1_out],dim=1))
        return self.out(torch.cat([de0,x],dim=1))
    

class cGAN_gen(nn.Module):  
    def __init__(self,in_channels,out_channels):  
        super(cGAN_gen, self).__init__()  
        self.encoder = rebuild_encoder(in_channels)  
        self.decoder = rebuild_decoder(out_channels)
    def forward(self,data,label):
        x,conv1_out,conv2_out,conv3_out,conv4_out=self.encoder(data,label)
        out=self.decoder(x,conv1_out,conv2_out,conv3_out,conv4_out)
        return out 
    
class cGAN_disc(nn.Module):  
    def __init__(self,in_channels):  
        super(cGAN_disc, self).__init__()  
        self.encoder = rebuild_encoder(in_channels)
        self.down1 = nn.Sequential(
            conv2d_bn_relu(16,4,6,2,2),
            nn.MaxPool2d(4,4),
            conv2d_bn_relu(4,4,3,1,1),
        )
        self.down2 = nn.Sequential(
            conv2d_bn_relu(64,8,6,2,2),
            nn.MaxPool2d(2,2),
            conv2d_bn_relu(8,8,3,1,1),
        )
        self.down3 = nn.Sequential(
            deconv_relu(256,16,6,2,2),
            nn.MaxPool2d(2,2),
            conv2d_bn_relu(16,16,3,1,1),
            nn.MaxPool2d(2,2)
        )  
        self.down4 = nn.Sequential(
            deconv_relu(512,200,6,2,2),
            nn.MaxPool2d(2,2),
            deconv_relu(200,100,6,2,2),
            nn.MaxPool2d(2,2),
            deconv_relu(100,50,6,2,2),
            nn.MaxPool2d(2,2),
        )
        self.out=nn.Sequential(
            nn.Linear(78*16*16,1200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1200,64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64,1),   
            nn.Sigmoid()     
        )
    def forward(self,data,label):
        x,conv1_out,conv2_out,conv3_out,conv4_out=self.encoder(data,label)
        down1=self.down1(conv1_out)
        down2=self.down2(conv2_out)
        down3=self.down3(conv3_out)
        down4=self.down4(conv4_out)
        down=torch.cat([down1,down2,down3,down4],dim=1)
        down=down.view(-1,78*16*16)
        out=self.out(down)
        return out 

    
