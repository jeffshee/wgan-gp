import os
import torch
from torch.autograd import grad as torch_grad
from torchvision.utils import make_grid, save_image


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=50, save_every=10, writer=None,
                 use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.save_every = save_every
        self.writer = writer

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

        """
        Create output folders
        imgs -> output images (generated samples etc.)
        saves -> state of model that saved every 'save_every' epoch
        finals -> final (end of train) state of model
        """
        self.root_out = '../output/'
        for path in ['imgs', 'saves', 'finals']:
            try:
                os.makedirs(self.root_out + path)
            except OSError:
                pass

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        one = torch.ones(1)
        if self.use_cuda:
            alpha = alpha.cuda()
            one = one.cuda()
        interpolated = alpha * real_data.data + (one - alpha) * generated_data.data
        interpolated.requires_grad_()
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader, epoch, epochs):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data[0])
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data[0])

            if (i + 1) % self.print_every == 0:
                print('Epoch: [{}/{}] Iteration: [{}/{}] D: {:.3f}, G: {:.3f}, GP: {:.3f}, GradNorm: {:.3f}'
                      .format(epoch + 1, epochs, i + 1, len(data_loader), self.losses['D'][-1],
                              self.losses['G'][-1] if self.num_steps > self.critic_iterations else 0,
                              self.losses['GP'][-1], self.losses['gradient_norm'][-1]))

    def train(self, data_loader, epochs):
        # Fix latents to see how image generation improves during training
        fixed_latents = self.G.sample_latent(64)
        if self.use_cuda:
            fixed_latents = fixed_latents.cuda()

        fixed_real = next(iter(data_loader))[0]
        if self.use_cuda:
            fixed_real = fixed_real.cuda()

        img = make_grid(fixed_real, normalize=True)
        save_image(img, self.root_out + 'imgs/fixed_real.png')
        if self.writer is not None:
            self.writer.add_image('Real/Sample', img)

        for epoch in range(epochs):
            self._train_epoch(data_loader, epoch, epochs)

            # Save states
            if (epoch + 1) % self.save_every == 0:
                torch.save(self.G.state_dict(), self.root_out + 'saves/gen_epoch_{}.pth'.format(epoch + 1))
                torch.save(self.D.state_dict(), self.root_out + 'saves/dis_epoch_{}.pth'.format(epoch + 1))

            # Generate batch of images and convert to grid
            img = make_grid(self.G(fixed_latents).cpu().data, normalize=True)
            save_image(img, self.root_out + 'imgs/training_epoch_{}.png'.format(epoch + 1))
            if self.writer is not None:
                self.writer.add_image('Generated/Sample', img, epoch)

        # Save final states
        torch.save(self.G.state_dict(), self.root_out + 'finals/gen_final.pth')
        torch.save(self.G.state_dict(), self.root_out + 'finals/gen_final.pth')

        if self.writer is not None:
            self.writer.close()

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
