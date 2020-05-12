import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

from PIL import Image
import cv2
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        latent_size = 16
        self.latent_size = latent_size

        self.fc1 = nn.Linear(32*32, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)
        self.fc3 = nn.Linear(latent_size, 400)
        self.fc4 = nn.Linear(400, 32*32)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def to_latent(self, x):
        x = x.view(-1, 32*32)
        x = F.relu(self.fc1(x))
        z = F.relu(self.fc21(x))
        return z
    
    def from_latent(self, z):
        x = F.relu(self.fc3(z))
        x = F.sigmoid(self.fc4(x))
        return x

    def forward(self, x):
        x = x.view(-1, 32*32)
        x = F.relu(self.fc1(x))
        z = F.relu(self.fc21(x))
        x = F.relu(self.fc3(z))
        x = torch.sigmoid(self.fc4(x))

        return x, z, z


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_size = 16
        self.latent_size = latent_size

        self.fc1 = nn.Linear(32*32, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)
        self.fc3 = nn.Linear(latent_size, 400)
        self.fc4 = nn.Linear(400, 32*32)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def to_latent(self, x):
        mu, logvar = self.encode(x.view(-1, 32*32))
        z = self.reparameterize(mu, logvar)
        return z
    
    def from_latent(self, z):
        return self.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 32*32))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ShapeEncoder:
    def __init__(self):
        self.model_ae = AutoEncoder()
        self.model_ae.load_state_dict(torch.load('./ae_best_16_size_32.pth'))
        self.model_ae = self.model_ae.cpu()

        self.model_bvae = VAE()
        self.model_bvae.load_state_dict(torch.load('./best_16_size_32.pth'))
        self.model_bvae = self.model_bvae.cpu()

    def encode(self, model, img):
        img = img[:,:]/255
        img = torch.tensor(img, dtype=torch.float).unsqueeze(0)
        return model.to_latent(img).detach().numpy()[0]

    def encode_bvae(self, img):
        return self.encode(self.model_bvae, img)

    def encode_ae(self, img):
        return self.encode(self.model_ae, img)

    def decode_bvae(self, z, threshold=None ):
        return self.decode(self.model_bvae, z, threshold)

    def decode_ae(self, z, threshold=None ):
        return self.decode(self.model_ae, z, threshold)

    def decode(self, model, z, threshold=None):
        z = torch.tensor(z, dtype=torch.float).unsqueeze(0)
        result = model.from_latent(z)[0].detach().numpy()
        result *= 255
        result = np.clip(result,0,255).astype('uint8')
        
        if threshold is not None:
            result = (result>=threshold).astype('uint8') * 255
        
        result = result.reshape(input_image_size, input_image_size,1)
        #result = np.repeat(result, 3, axis=-1)
        return result