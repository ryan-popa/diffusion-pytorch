import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE


def loss_function(recon_x, x, mu, logvar):
    # BCE makes sure the VAE reconstructs the images well.
    # KLD ensures the latent space has a structure similar to a Gaussian distribution, making it useful for generative tasks.
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training hyperparameters
BATCH_SIZE = 256
EPOCHS = 100
LR = 0.001

# Load data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, optimizer, and move to device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
print(f"Using device: {device}")
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item()/len(data):.6f}")
    
    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

print("Training completed!")

torch.save(model.state_dict(), 'vae_model.pth')
print("Model saved to vae_model.pth")
