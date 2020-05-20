import logging

import torch
import torchvision

from convNEAT import ConvNEAT


def mnist(torch_device):
    # Build datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
            lambda x: x.to(device=torch_device)),
        torchvision.transforms.Normalize((1 / 2,), (1 / 2,)),
    ])
    target_transform = torchvision.transforms.Lambda(lambda x: torch.tensor(x, device=torch_device))
    data_train = torchvision.datasets.MNIST(
        'data', train=True, transform=transform,
        target_transform=target_transform, download=True)
    data_test = torchvision.datasets.MNIST(
        'data', train=False, transform=transform,
        target_transform=target_transform, download=True)
    return data_train, data_test


if __name__ == '__main__':
    # Debugging
    logging.basicConfig(level='INFO')
    # Train on GPU
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # The dataset
    data_train, data_test = mnist(torch_device)

    # train
    trainer = ConvNEAT(output_size=10, n=20, torch_device=torch_device, name='test_run_3', seed=20)
    trainer.prompt(data_train, save_mode="elites", elitism_rate=0.3, min_species_size=4, epochs=3,
                   n_generations_no_change=3, tol=0, min_species=1, mutate_speed=0.3)
