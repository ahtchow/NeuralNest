import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from model.ViT import ViT
import tests

np.random.seed(0)
torch.manual_seed(0)

def run_tests():
    tests.test_patchify()
    tests.test_linear_projection()
    tests.test_add_class_embedding()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    print("\nRunning Tests:")
    run_tests()
    print("\033\033[0m")

    config = {'H': 28, 'W': 28, 'C': 1, 'P': 7, 'LP': 8, 
              'num_head': 2, 'num_enc_blocks': 3, 'mlp_ratio': 4, 'out_dim': 10}

    N_EPOCHS = 10
    LR = 0.001
    BATCH_SIZE = 64

    transform = ToTensor()
    train_set = MNIST(root='./datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./datasets', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)

    model = ViT(config, verbose=True)
    if device != "cpu":
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            if device != "cpu": 
                x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            if device != "cpu": 
                x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

if __name__ == "__main__":
    main()