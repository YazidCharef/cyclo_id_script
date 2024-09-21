from models import initialize_stormnetc1, initialize_resnet
from data_processing import load_and_preprocess_data
from train import train
from evaluate import evaluate
import config

def main():
    print(f"Using device: {config.DEVICE}")

    # Define models and configurations
    experiments = [
        (initialize_stormnetc1, "StormNETC1", False),
        (initialize_stormnetc1, "StormNETC1", True)

        (initialize_resnet, "ResNet", False),
        (initialize_resnet, "ResNet", True),
    ]

    # Run experiments
    for model_init, model_name, use_sst in experiments:
        print(f"\nTraining and evaluating {model_name} {'with' if use_sst else 'without'} SST...")

        # Load and preprocess data for each experiment
        train_loader, val_loader, test_loader = load_and_preprocess_data(
            netcdf_file=config.NETCDF_PATH,
            typhoon_file=config.TYPHOON_FILE,
            sst_file=config.SST_FILE,
            batch_size=config.BATCH_SIZE,
            start_year=config.START_YEAR,
            end_year=config.END_YEAR,
            use_sst=use_sst
        )
        
        model = model_init(use_sst=use_sst)

        model, train_losses, val_losses, train_accuracies = train(
            model, train_loader, val_loader, 
            config.NUM_EPOCHS, config.LEARNING_RATE, config.DEVICE, use_sst=use_sst
        )

        test_loss, f1_score, test_accuracy = evaluate(
            model, test_loader, config.DEVICE, use_sst, 
            train_losses, val_losses, train_accuracies
        )
        
        print(f"{model_name} {'with' if use_sst else 'without'} SST - Test Loss: {test_loss:.4f}, F1 Score: {f1_score:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()