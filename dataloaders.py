from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(batch_size=128, normalize=True):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    trans = transforms.Compose([transforms.Resize(32),
                                transforms.ToTensor()])
    if normalize:
        trans = transforms.Compose([trans, transforms.Normalize((0.5,), (0.5,))])
    # Get train and test data
    train_data = datasets.MNIST('../data', train=True, download=True, transform=trans)
    test_data = datasets.MNIST('../data', train=False, transform=trans)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128, normalize=True):
    """Fashion MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    trans = transforms.Compose([transforms.Resize(32),
                                transforms.ToTensor()])
    if normalize:
        trans = transforms.Compose([trans, transforms.Normalize((0.5,), (0.5,))])
    # Get train and test data
    train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True, transform=trans)
    test_data = datasets.FashionMNIST('../fashion_data', train=False, transform=trans)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_lsun_dataloader(dataset='bedroom_train', batch_size=64, normalize=True):
    """LSUN dataloader with (128, 128) sized images.

    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    trans = transforms.Compose([transforms.Resize(128),
                                transforms.CenterCrop(128),
                                transforms.ToTensor()])
    if normalize:
        trans = transforms.Compose([trans, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Get dataset
    lsun_dset = datasets.LSUN('../lsun', classes=[dataset], transform=trans)
    # Create dataloader
    return DataLoader(lsun_dset, batch_size=batch_size, shuffle=True)


def get_img_folder_dataloader(path_to_data, image_size=128, batch_size=128, normalize=True):
    trans = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor()])
    if normalize:
        trans = transforms.Compose([trans, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dset = datasets.ImageFolder(root=path_to_data, transform=trans)
    return DataLoader(dset, batch_size=batch_size, shuffle=True)
