import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np

bs = 100
# MNIST Dataset
train_dataset = datasets.MNIST(root='/kaggle/working/mnist-data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='/kaggle/working/mnist-data', train=False, transform=transforms.ToTensor(), download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# build model
vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
if torch.cuda.is_available():
    vae.cuda()
class VAE_CNN(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE_CNN, self).__init__()
        
        # encoder part
        # self.fc1 = nn.Linear(x_dim, h_dim1)
        # self.fc2 = nn.Linear(h_dim1, h_dim2)
        # self.fc31 = nn.Linear(h_dim2, z_dim)
        # self.fc32 = nn.Linear(h_dim2, z_dim)
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc3 = nn.Linear(32*7*7, h_dim1)
        self.fc4 = nn.Linear(h_dim1, h_dim2)
        self.fc51 = nn.Linear(h_dim2, z_dim)
        self.fc52 = nn.Linear(h_dim2, z_dim)

        # decoder part
        # self.fc4 = nn.Linear(z_dim, h_dim2)
        # self.fc5 = nn.Linear(h_dim2, h_dim1)
        # self.fc6 = nn.Linear(h_dim1, x_dim)
        self.fc6 = nn.Linear(z_dim, h_dim2)
        self.fc7 = nn.Linear(h_dim2, h_dim1)
        self.fc8 = nn.Linear(h_dim1, 32*7*7)
        self.tcnn9 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.tcnn10 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, output_padding=1, padding=1)
        
    def encoder(self, x):
        # h = F.relu(self.fc1(x))
        # h = F.relu(self.fc2(h))
        # return self.fc31(h), self.fc32(h) # mu, log_var
        h = F.relu(self.cnn1(x))
        h = F.relu(self.cnn2(h))
        h = F.relu(self.fc3(h.view(-1, 32*7*7)))
        h = F.relu(self.fc4(h))
        return self.fc51(h), self.fc52(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        # h = F.relu(self.fc4(z))
        # h = F.relu(self.fc5(h))
        # return F.sigmoid(self.fc6(h)) 
        h = F.relu(self.fc6(z))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h)).view(-1, 32, 7, 7)
        h = F.relu(self.tcnn9(h))
        return torch.sigmoid(self.tcnn10(h))
    
    def forward(self, x):
        # mu, log_var = self.encoder(x.view(-1, 784))
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# build model
vae = VAE_CNN(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
if torch.cuda.is_available():
    vae.cuda()

class VAE_CNN_Singular(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE_CNN_Singular, self).__init__()
        
        # encoder part
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        #self.cnn2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc3 = nn.Linear(32*14*14, h_dim1)
        self.fc4 = nn.Linear(h_dim1, h_dim2)
        self.fc51 = nn.Linear(h_dim2, z_dim)
        self.fc52 = nn.Linear(h_dim2, z_dim)

        # decoder part
        self.fc6 = nn.Linear(z_dim, h_dim2)
        self.fc7 = nn.Linear(h_dim2, h_dim1)
        self.fc8 = nn.Linear(h_dim1, 32*14*14)
        #self.tcnn9 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.tcnn10 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, output_padding=1, padding=1)
        
    def encoder(self, x):
        h = F.relu(self.cnn1(x))
        h = F.relu(self.cnn2(h))
        h = F.relu(self.fc3(h.view(-1, 32*7*7)))
        h = F.relu(self.fc4(h))
        return self.fc51(h), self.fc52(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc6(z))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h)).view(-1, 32, 7, 7)
        h = F.relu(self.tcnn9(h))
        return torch.sigmoid(self.tcnn10(h))
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# build model
vae_singular = VAE_CNN_Singular(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
if torch.cuda.is_available():
    vae.cuda()


optimizer = optim.Adam(vae.parameters())
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test():
    vae.eval()
    test_loss= 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)

            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, 51):
    train(epoch)
    test()


with torch.no_grad():
    z = torch.randn(64, 2).cuda()
    for i in range(2):
        z[:, i] = torch.linspace(-4,0, 64)

    sample = vae.decoder(z).cuda()
    
    save_image(sample.view(64, 1, 28, 28), '/kaggle/working/sample1_' + '.png')