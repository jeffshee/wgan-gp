# Wasserstein GAN with Gradient penalty

Pytorch implementation of [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) by Gulrajani et al.
Forked from https://github.com/EmilienDupont/wgan-gp, with some corrections and improvements, for instance:
1. Discriminator's output for WGAN-GP doesn't necessarily have to be 0~1, hence removed the Sigmoid()
2. Also added some Generator and Discriminator.
3. Removed Gif functionality. Perhaps a video clip is a better choice.
4. Updated some codes for recent PyTorch.
5. TensorBoard support.