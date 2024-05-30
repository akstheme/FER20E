import os
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image  # Add this import statement

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print("Number of available GPUs:", num_gpus)
#os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define data paths
data_dir = '/home/kuldeep/FER20E/'

# Get the list of subfolders (labels) in the dataset directory
labels = [label for label in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, label))]

# Create a dictionary to map labels to indices
label_to_idx = {label: idx for idx, label in enumerate(labels)}

# Create a list to hold the paths of all images in the dataset
all_image_paths = []

for label in labels:
    label_path = os.path.join(data_dir, label)
    image_paths = [os.path.join(label_path, image) for image in os.listdir(label_path)]
    all_image_paths.extend(image_paths)

# Split the dataset paths into training and validation sets
train_paths, test_paths = train_test_split(all_image_paths, test_size=0.3, random_state=42)
train_paths, val_paths = train_test_split(train_paths, test_size=0.3, random_state=42)

# Define parameters
num_classes = 20
batch_size = 32
learning_rate = 1e-4
num_epochs = 10

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, file_paths, label_to_idx, transform=None):
        self.file_paths = file_paths
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = os.path.basename(os.path.dirname(image_path))  # Extract label from the parent folder
        label_idx = self.label_to_idx[label]  # Get the index of the label from the dictionary
        return image, label_idx  # Return label index instead of label string

# Create custom datasets
train_dataset = CustomDataset(train_paths, label_to_idx, transform=transform)
val_dataset = CustomDataset(val_paths, label_to_idx, transform=transform)
test_dataset = CustomDataset(test_paths, label_to_idx, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained ViT model and feature extractor
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)

# Check GPU availability

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:3'
if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Currently selected GPU: {torch.cuda.current_device()}")
else:
    print("No GPUs available.")


model.to(device)
#device = 'cuda:3'
# Define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        train_loss += loss.item() * inputs.size(0)
    
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    train_loss = train_loss / len(train_dataset)
    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

print('Training complete.')

# Save the trained model
output_dir = '/DATA/kuldeep/My_Research/ViT_models1/'
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
