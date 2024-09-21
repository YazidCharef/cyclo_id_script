import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(preds, labels, num_classes):

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        confusion_matrix[t, p] += 1

    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    f1_scores = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

        precisions[i] = true_positives / (true_positives + false_positives + 1e-7)
        recalls[i] = true_positives / (true_positives + false_negatives + 1e-7)
        f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i] + 1e-7)

    class_counts = np.sum(confusion_matrix, axis=1)
    weights = class_counts / np.sum(class_counts)
    f1_global = np.sum(f1_scores * weights)

    return f1_global, f1_scores, accuracy

def evaluate_model(model, test_loader, device, num_classes=4):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    f1_global, f1_per_class, accuracy = calculate_metrics(np.array(all_preds), np.array(all_labels), num_classes)
    
    return avg_loss, f1_global, f1_per_class, accuracy, all_preds, all_labels

def plot_confusion_matrix(true_labels, pred_labels, class_names, filename):
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_training_progress(train_losses, val_losses, train_accuracies, test_metrics, model_name, use_sst):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(20, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name} {"with" if use_sst else "without"} SST - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, 'g-', label='Training Accuracy')
    plt.title(f'{model_name} {"with" if use_sst else "without"} SST - Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot test metrics
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.text(0.5, 0.8, f'Test Loss: {test_metrics["loss"]:.4f}', horizontalalignment='center')
    plt.text(0.5, 0.6, f'Test Accuracy: {test_metrics["accuracy"]:.4f}', horizontalalignment='center')
    plt.text(0.5, 0.4, f'Global F1 Score: {test_metrics["f1_global"]:.4f}', horizontalalignment='center')
    plt.title(f'{model_name} {"with" if use_sst else "without"} SST - Test Metrics')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_{"with" if use_sst else "without"}_SST_training_progress.png')
    plt.close()

    print(f"Training progress plot saved as {model_name}_{'with' if use_sst else 'without'}_SST_training_progress.png")

def evaluate(model, test_loader, device, use_sst, train_losses, val_losses, train_accuracies):
    model_name = model.__class__.__name__
    sst_status = "with_sst" if use_sst else "without_sst"
    
    print(f"Evaluating {model_name} ({sst_status})...")
    avg_loss, f1_global, f1_per_class, accuracy, all_preds, all_labels = evaluate_model(model, test_loader, device)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Global F1 Score: {f1_global:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    class_names = ['No Cyclone', 'Cyclogenesis', 'Full Typhoon', 'Cyclolysis']
    for i, class_name in enumerate(class_names):
        print(f"F1 Score for {class_name}: {f1_per_class[i]:.4f}")
    
    cm_filename = f"confusion_matrix_{model_name}_{sst_status}.png"
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_filename)
    print(f"Confusion matrix saved as '{cm_filename}'")
    
    # Plot training progress
    test_metrics = {"loss": avg_loss, "accuracy": accuracy, "f1_global": f1_global}
    plot_training_progress(train_losses, val_losses, train_accuracies, test_metrics, model_name, use_sst)
    
    # Save evaluation results to a text file
    results_filename = f"evaluation_results_{model_name}_{sst_status}.txt"
    with open(results_filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"SST Status: {'Used' if use_sst else 'Not Used'}\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Global F1 Score: {f1_global:.4f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final Training Accuracy: {train_accuracies[-1]:.4f}\n")
        for i, class_name in enumerate(class_names):
            f.write(f"F1 Score for {class_name}: {f1_per_class[i]:.4f}\n")
    print(f"Evaluation results saved to '{results_filename}'")
    
    return avg_loss, f1_global, accuracy