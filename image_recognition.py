import torch
from torchvision import transforms
from PIL import Image


def image_recognition(image_path):
    model = torch.load('image_recognition.pth')

    model.eval()

    input_image_transformation = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])

    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    image = input_image_transformation(image)
    
    # Add a batch dimension
    image = image.unsqueeze(0)

    image = image.cuda()
    
    # Run the model on the input image
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    # Load the class labels
    with open('food-101\\food-101\\food-101\\meta\\classes.txt', 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]

    predicted_class_label = class_labels[preds.item()]

    predicted_class_label = predicted_class_label.replace('_', ' ')

    return predicted_class_label

