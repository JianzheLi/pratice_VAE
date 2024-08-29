import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

bs = 100

# MNIST数据集
train_dataset = datasets.MNIST(root='/kaggle/working/mnist-data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='/kaggle/working/mnist-data', train=False, transform=transforms.ToTensor(), download=True)

# 数据加载器（输入管道）
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

class VAE(nn.Module):
    def __init__(self, x_dim,z_dim):
        super(VAE, self).__init__()
        
        # 编码器部分
        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc31 = nn.Linear(32, z_dim)
        self.fc32 = nn.Linear(32, z_dim)

        self.b1=nn.BatchNorm1d(512)
        self.b2=nn.BatchNorm1d(256)
        self.b3=nn.BatchNorm1d(128)
        self.b4=nn.BatchNorm1d(64)
        self.b5=nn.BatchNorm1d(32)

        

        
        # 解码器部分
        self.fc6 = nn.Linear(z_dim, 128)
        self.fc7 = nn.Linear(128, 512)
        self.fc8 = nn.Linear(512, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.b1(self.fc1(x)))
        h = F.relu(self.b2(self.fc2(h)))
        h = F.relu(self.b3(self.fc3(h)))
        h = F.relu(self.b4(self.fc4(h)))
        h = F.relu(self.b5(self.fc5(h)))
        z1=self.fc31(h)
        z2=self.fc32(h)  # mu, log_var
        return z1,z2
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # 返回z样本
        
    def decoder(self, z):
        h = F.relu(self.fc6(z))
        h = F.relu(self.fc7(h))
        return torch.sigmoid(self.fc8(h))
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# 构建模型
vae = VAE(x_dim=784,z_dim=2)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters())

# 损失函数（重构损失 + KL散度）
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test():
    vae.eval()
    test_loss = 0
    z_min = torch.tensor([float('inf')] * 2).cuda()
    z_max = torch.tensor([float('-inf')] * 2).cuda()
    
    with torch.no_grad():
        for data, _ in test_loader:
            if torch.cuda.is_available():
                data = data.cuda()
            recon, mu, log_var = vae(data)
            
            # 计算z
            z = vae.sampling(mu, log_var)
            
            # 更新z向量的最小值和最大值
            z_min = torch.minimum(z_min, z.min(dim=0).values)
            z_max = torch.maximum(z_max, z.max(dim=0).values)
            
            # 累加批量损失
            test_loss += loss_function(recon, data, mu, log_var).item()
    
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    
    print(f'z vector min values: {z_min.cpu().numpy()}')
    print(f'z vector max values: {z_max.cpu().numpy()}')
for epoch in range(1, 51):
    train(epoch)
    test()

with torch.no_grad():
    z = torch.randn(64, 2).cuda()
    for i in range(2):
        z[:, i] = torch.linspace(-5, 0, 64)

    sample = vae.decoder(z).cuda()
    
    save_image(sample.view(64, 1, 28, 28), '/kaggle/working/sample1.png')
