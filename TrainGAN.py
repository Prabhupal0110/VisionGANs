import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = datasets.CelebA(root='./data', download=True, transform=transform)

# dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)



import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),  # Input: (batch_size, 100, 1, 1) -> Output: (batch_size, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # Output: (batch_size, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # Output: (batch_size, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # (batch_size, 64, 16, 16) -> (batch_size, 3, 32, 32)
            nn.Tanh()  # Output pixel values between -1 and 1

        )

    def forward(self, x):
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # Input: (batch_size, 3, 32, 32) -> Output: (batch_size, 64, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output: (batch_size, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output: (batch_size, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 2, 1, bias=False),  # Output: (batch_size, 1, 2, 2)
            nn.Sigmoid()  # Output probability between 0 and 1 (real or fake)
        )

    def forward(self, x):
        x = self.model(x)
        # x = x.view(256, -1).mean(1, keepdim=True)  # Flatten to (batch_size, 1)
        return x


import torch.optim as optim
import torch


if __name__ == "__main__":

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(128, 100, 1, 1).cuda()

    for epoch in range(2):
        for i, (data, _) in enumerate(dataloader):
            real_images = data.cuda()
            batch_size = real_images.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(batch_size, 1).cuda()
            fake_labels = torch.zeros(batch_size, 1).cuda()
            # print(real_images.shape, real_labels.shape, fake_labels.shape )


            outputs = discriminator(real_images)
            outputs = outputs.view(batch_size, -1).mean(1, keepdim=True)  # Flatten to (batch_size, 1)
            loss_real = criterion(outputs, real_labels)
            loss_real.backward(retain_graph=True)

            noise = torch.randn(batch_size, 100, 1, 1).cuda()
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            outputs = outputs.view(batch_size, -1).mean(1, keepdim=True)  # Flatten to (batch_size, 1)
            loss_fake = criterion(outputs, fake_labels)
            loss_fake.backward(retain_graph=True)

            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            outputs = outputs.view(batch_size, -1).mean(1, keepdim=True)  # Flatten to (batch_size, 1)
            loss_g = criterion(outputs, real_labels)
            loss_g.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/100], Loss D: {loss_real + loss_fake}, Loss G: {loss_g}")

    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu()
        plt.figure(figsize=(20,20))
        plt.imshow(torchvision.utils.make_grid(fake_images, normalize=True).permute(1, 2, 0))
        plt.show()