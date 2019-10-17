import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders
from models import Generator, Discriminator
from training import Trainer
# TensorBoard support (optional)
from torch.utils.tensorboard import SummaryWriter

data_loader, _ = get_mnist_dataloaders(batch_size=64)
img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 200
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  writer=SummaryWriter('runs'), use_cuda=torch.cuda.is_available())

# Load Model
# trainer.G.load_state_dict(torch.load('../output/finals/gen_final.pth'))
# trainer.D.load_state_dict(torch.load('../output/finals/dis_final.pth'))

trainer.train(data_loader, epochs)


