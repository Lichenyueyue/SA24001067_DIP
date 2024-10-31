import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.dconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.dconv4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.encoder=nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4
        )
        self.decoder=nn.Sequential(
            self.dconv1,
            self.dconv2,
            self.dconv3,
            self.dconv4
        )

    def forward(self, x):
        x=self.encoder(x)

        output = self.decoder(x)
        
        return output
    