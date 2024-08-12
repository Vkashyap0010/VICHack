import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch

class GroceryDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        label = self.annotations[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create the dataset and dataloader
img_dir = 'path/to/grocery/images'
annotations_file = 'path/to/grocery/annotations.json'
grocery_dataset = GroceryDataset(img_dir, annotations_file, transform=transform)
grocery_dataloader = DataLoader(grocery_dataset, batch_size=32, shuffle=True, num_workers=4)

# Load a pre-trained ResNet model
ingredient_model = models.resnet18(pretrained=True)
num_ftrs = ingredient_model.fc.in_features
ingredient_model.fc = nn.Linear(num_ftrs, len(grocery_dataset.annotations[0]['label']))  # Number of unique labels

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ingredient_model = ingredient_model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ingredient_model.parameters(), lr=0.001)

# Training loop
def train_ingredient_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item() * images.size(0)
        
        # Print statistics
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    
    return model

# Train the model
ingredient_model = train_ingredient_model(ingredient_model, grocery_dataloader, criterion, optimizer)