import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch

class RecipeDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        with open(data_file, 'r') as f:
            self.recipes = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.recipes)
    
    def __getitem__(self, idx):
        recipe = self.recipes[idx]
        ingredients = recipe['ingredients']
        instructions = recipe['instructions']
        
        # Tokenize the ingredients and instructions
        ingredients_text = ' '.join(ingredients)
        instructions_text = ' '.join(instructions)
        
        inputs = self.tokenizer(ingredients_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        outputs = self.tokenizer(instructions_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        
        return inputs.input_ids.squeeze(), outputs.input_ids.squeeze()

# Load the tokenizer
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the dataset and dataloader
recipe_data_file = 'path/to/recipe1m.json'
recipe_dataset = RecipeDataset(recipe_data_file, tokenizer)
recipe_dataloader = DataLoader(recipe_dataset, batch_size=16, shuffle=True, num_workers=4)


from transformers import GPT2LMHeadModel

# Load the pre-trained GPT-2 model
recipe_model = GPT2LMHeadModel.from_pretrained('gpt2')
recipe_model = recipe_model.to(device)


# Define the optimizer
optimizer = torch.optim.Adam(recipe_model.parameters(), lr=1e-5)

# Training loop
def train_recipe_model(model, dataloader, optimizer, num_epochs=5):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, outputs in dataloader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs_pred = model(inputs, labels=outputs)
            loss = outputs_pred.loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item() * inputs.size(0)
        
        # Print statistics
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
    
    return model

# Train the model
recipe_model = train_recipe_model(recipe_model, recipe_dataloader, optimizer)


def generate_recipe(ingredient_model, recipe_model, image, tokenizer, max_length=512):
    ingredient_model.eval()
    recipe_model.eval()
    
    # Transform the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict the ingredients
    with torch.no_grad():
        outputs = ingredient_model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Get the ingredient labels
    ingredients = [grocery_dataset.annotations[i]['label'] for i in predicted]
    ingredients_text = ' '.join(ingredients)
    
    # Generate the recipe
    inputs = tokenizer(ingredients_text, return_tensors='pt').input_ids.to(device)
    recipe_ids = recipe_model.generate(inputs, max_length=max_length, num_beams=5, early_stopping=True)
    recipe = tokenizer.decode(recipe_ids[0], skip_special_tokens=True)
    
    return ingredients, recipe

# Example usage
from PIL import Image

image_path = 'path/to/test/image.jpg'
image = Image.open(image_path).convert('RGB')

ingredients, recipe = generate_recipe(ingredient_model, recipe_model, image, tokenizer)
print("Recognized Ingredients:", ingredients)
print("Generated Recipe:", recipe)