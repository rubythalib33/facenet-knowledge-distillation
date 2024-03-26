import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
import models
from config import MODEL, DATA_SOURCE, EPOCHS, LR, OPTIMIZER, BATCH_SIZE
import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Augmentations: auto rotate, auto horizontal flip, and auto brightness/contrast
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),  # Auto horizontal flip
    transforms.RandomRotation(20),  # Auto rotate +/- 20 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Auto brightness/contrast
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(DATA_SOURCE, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize teacher and student models
teacher_model = InceptionResnetV1(pretrained='vggface2').eval()
student_model = models.get_model(MODEL)

# Move models to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
if OPTIMIZER == 'adam':
    optimizer = optim.Adam(student_model.parameters(), lr=LR)
elif OPTIMIZER == 'sgd':
    optimizer = optim.SGD(student_model.parameters(), lr=LR, momentum=0.9)
else:
    raise ValueError("Unsupported optimizer")

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Function to save model
def save_model(model, model_name, optimizer_name, epoch):
    os.makedirs("ckpt", exist_ok=True)  # Create ckpt directory if it doesn't exist
    file_name = f"{model_name}_{optimizer_name}.pt"
    save_path = os.path.join("ckpt", file_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Plot and save training losses
def plot_and_save_losses(losses, model):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join("plots", f"training_loss_{model}.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

# Training loop
def train(dataloader, teacher_model, student_model, criterion, optimizer, scheduler, epochs):
    losses = []
    student_model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_loss = 0.0
        for images, _ in tqdm(dataloader, desc="Batches", leave=False):
            images = images.to(device)

            # Forward pass through teacher and student models
            with torch.no_grad():
                teacher_output = teacher_model(images)
            student_output = student_model(images)

            # Calculate loss and perform backprop
            loss = criterion(student_output, teacher_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        # Step the scheduler
        scheduler.step(epoch_loss)

        # Save model periodically or based on your criteria
        save_model(student_model, MODEL, OPTIMIZER, epoch + 1)

    plot_and_save_losses(losses, MODEL)

if __name__ == "__main__":
    train(dataloader, teacher_model, student_model, criterion, optimizer, scheduler, EPOCHS)
