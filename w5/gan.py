import sys
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

print(sys.version) # python 3.6
print(torch.__version__) # 1.0.1

def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0, 2).transpose(0, 1) # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x.view(x.size(0), x.size(1), 1, 1))

D = Discriminator()
print(D)
G = Generator()
print(G)

dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/',
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))]),
                       download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

ix = 149
x, _ = dataset[ix]
plt.matshow(x.squeeze().numpy(), cmap=plt.cm.gray)
plt.colorbar()

Dscore = D(x.unsqueeze(0))
Dscore

xbatch, _ = next(iter(dataloader)) # 64 x 1 x 28 x 28: minibatch of 64 samples
xbatch.shape
D(xbatch) # 64x1 tensor: 64 predictions of probability of input being real.
D(xbatch).shape

show_imgs(xbatch)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)
D = Discriminator().to(device)
G = Generator().to(device)
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

collect_x_gen = []
fixed_noise = torch.randn(64, 100, device=device)
fig = plt.figure() # keep updating this one
plt.ion()

losses = {'G': [], 'D': []}

for epoch in range(3): # 3 epochs
    for i, data in enumerate(dataloader, 0):
        x_real, _ = data
        x_real = x_real.to(device)
        b_size = x_real.size(0)  # 获取当前批次大小
        lab_real = torch.ones(b_size, device=device)
        lab_fake = torch.zeros(b_size, device=device)
        optimizerD.zero_grad()

        D_x = D(x_real)
        lossD_real = criterion(D_x, lab_real)

        z = torch.randn(b_size, 100, device=device) # random noise, 当前批次大小, z_dim=100
        x_gen = G(z).detach()
        D_G_z = D(x_gen)
        lossD_fake = criterion(D_G_z, lab_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
        
        optimizerG.zero_grad()

        z = torch.randn(b_size, 100, device=device) # random noise, 当前批次大小, z_dim=100
        x_gen = G(z)
        D_G_z = D(x_gen)
        lossG = criterion(D_G_z, lab_real) # -log D(G(z))

        lossG.backward()
        optimizerG.step()

        losses['D'].append(lossD.item())
        losses['G'].append(lossG.item())

        if i % 100 == 0:
            x_gen = G(fixed_noise)
            show_imgs(x_gen, new_fig=False)
            fig.canvas.draw()
            print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                epoch, i, len(dataloader), D_x.mean().item(), D_G_z.mean().item()))
    x_gen = G(fixed_noise)
    collect_x_gen.append(x_gen.detach().clone())

for x_gen in collect_x_gen:
    show_imgs(x_gen)

plt.figure()
plt.plot(losses['D'], label='Discriminator')
plt.plot(losses['G'], label='Generator')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curves')
plt.show()

torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')

def generate_images(generator, z):
    with torch.no_grad():
        generated_images = generator(z).cpu()
    grid = vutils.make_grid(generated_images, nrow=8, normalize=True, pad_value=0.3)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig('generated_images.png')

generator = Generator().cpu()
generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))

custom_random_numbers = torch.randn(100, 100)
selected_indices = np.random.choice(range(100), 5, replace=False)
selected_random_numbers = custom_random_numbers[selected_indices]

for i, z in enumerate(selected_random_numbers):
    for j in range(3):
        adjusted_z = z + torch.randn_like(z) * 0.1  # 调整随机数
        print(f"Generating images for selected random number {i+1}, adjustment {j+1}")
        generate_images(generator, adjusted_z.unsqueeze(0).repeat(8, 1))
