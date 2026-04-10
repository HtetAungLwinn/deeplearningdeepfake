from torch.utils.data import Dataset, Subset
import torch
from torchvision import datasets, transforms, models

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor()
])

class RelabelDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.label

def make_subset(folder, label, n):
    dataset = datasets.ImageFolder(root=folder, transform=transform)
    indices = torch.randperm(len(dataset))[:n].tolist()
    return RelabelDataset(Subset(dataset, indices), label)



def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss / len(train_loader):.4f} | Val Acc: {val_accuracy:.2f}%")


def test(model, test_loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test accuracy: {100 * correct / total:.2f}%")