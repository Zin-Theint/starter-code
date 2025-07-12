import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from preprocess import get_test_transforms
import torch.nn as nn

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test data
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=get_test_transforms(), download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the same model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(device)
model.load_state_dict(torch.load('./models/cnn.pth'))
model.eval()

# Evaluate
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
