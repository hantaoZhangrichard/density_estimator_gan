import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
train_ds = torchvision.datasets.MNIST('data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, noise):
        img = self.main(noise)
        img = img.view(-1, 1, 28, 28)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(-1, 28*28)
        out = self.main(img)
        return out

generator = Generator()
discriminator = Discriminator()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

loss_fn = torch.nn.BCELoss()

def img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis('off')
    plt.show()

test_input = torch.randn(16, 100)

D_loss = []
G_loss = []

for epoch in range(5):
    d_epoch = 0
    g_epoch = 0
    count = len(dataloader)
    for step, (img, _) in enumerate(dataloader):
        size = img.size(0)
        random_noise = torch.randn(size, 100)
        optimizer_D.zero_grad()
        real_output = discriminator(img)
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))
        d_real_loss.backward()

        gen_img = generator(random_noise)
        fake_output = discriminator(gen_img.detach())
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()

        d_loss = d_fake_loss + d_real_loss
        optimizer_D.step()

        optimizer_G.zero_grad()
        fake_output = discriminator(gen_img)
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output))
        g_loss.backward()

        optimizer_G.step()

        with torch.no_grad():
            d_epoch += d_loss
            g_epoch += g_loss
    with torch.no_grad():
       d_epoch /= count
       g_epoch /= count
       D_loss.append(d_epoch)
       G_loss.append(g_epoch)
       print('Epoch:', epoch)
img_plot(generator, test_input)


