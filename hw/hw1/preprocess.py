import torch as th
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = th.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = th.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print("Number of training samples:", len(train_dataset))
print("Number of testing samples:", len(test_dataset))