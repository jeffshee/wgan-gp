import os
import torch
import torch.optim as optim
from dataloaders import get_mnist_dataloaders, get_lsun_dataloader, get_img_folder_dataloader
from models import Generator, Discriminator, GeneratorTanh, DiscriminatorInstanceNorm
from training import Trainer

# data_loader, _ = get_mnist_dataloaders(batch_size=64)
# img_size = (32, 32, 1)
#
# generator = Generator(img_size=img_size, latent_dim=100, dim=16)
# discriminator = Discriminator(img_size=img_size, dim=16)
#
# print(generator)
# print(discriminator)
#
# # Initialize optimizers
# lr = 1e-4
# betas = (.9, .99)
# G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
# D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
#
# # Train model
# epochs = 200
# trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
#                   use_cuda=torch.cuda.is_available())
# trainer.train(data_loader, epochs, save_training_gif=True)

data_loader = get_img_folder_dataloader('../animeface-character-dataset/thumb', image_size=128, batch_size=64)
img_size = (128, 128, 3)

generator = GeneratorTanh(img_size=img_size, latent_dim=100, dim=16)
discriminator = DiscriminatorInstanceNorm(img_size=img_size, dim=16)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 1000
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())

# Load Model
trainer.G.load_state_dict(torch.load('./output/finals/gen_final.pth'))
trainer.D.load_state_dict(torch.load('./output/finals/dis_final.pth'))

trainer.train(data_loader, epochs)
