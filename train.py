import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None, use_sst=True):
    best_val_loss = float('inf')
    train_losses, val_losses, train_accuracies = [], [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

        # Validation phase

        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_loss)

        print(f"Val Loss: {epoch_loss:.4f}")

        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            sst_info = "with_sst" if use_sst else "without_sst"
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}_{sst_info}_model.pth")
            print(f"New best model saved with loss: {best_val_loss:.4f}")

        if scheduler:
            scheduler.step(epoch_loss)

    return model, train_losses, val_losses, train_accuracies

def train(model, train_loader, val_loader, num_epochs, learning_rate, device, use_sst=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    print(f"Starting training for {model.__class__.__name__} {'with' if use_sst else 'without'} SST...")
    model, train_losses, val_losses, train_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler, use_sst
    )

    return model, train_losses, val_losses, train_accuracies

