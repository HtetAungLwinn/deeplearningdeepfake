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



def train(model, data_loader, valid_loader, criterion, optimizer, device, num_epochs=5, lr=0.001):
    for epoch in range(num_epochs):
        model.train()
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > 0.5).float

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%")
            
def test(model, test_loader, device):
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

         
        with torch.no_grad():
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float   
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        test_accuracy = 100 * correct / total  
        print(f"Test accuracy: {test_accuracy:.2f}")