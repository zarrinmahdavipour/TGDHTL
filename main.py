import argparse
import torch
import scipy.io as sio
from config import Config
from train import train_tgdhtl, TGDHTL
from utils import load_data, evaluate_model
from visualizations import plot_classification_map, plot_confusion_matrix

def main(args):
    """
    Main entry point for TGDHTL experiments.
    Supports training, testing, and visualization modes.
    """
    # Load configuration
    cfg = Config
    cfg.create_dirs()

    # Load data
    hsi_cube, labels = load_data(cfg.DATA_PATH, cfg.LABEL_PATH)

    # Initialize model
    model = TGDHTL(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)

    if args.mode == 'train':
        # Train model
        model = train_tgdhtl(
            hsi_cube=hsi_cube,
            labels=labels,
            num_epochs=cfg.NUM_EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            lr=cfg.LEARNING_RATE
        )
        torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
        print(f'Model saved to {cfg.MODEL_SAVE_PATH}')

    elif args.mode == 'test':
        # Load trained model
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH))
        model.eval()
        # Evaluate model
        accuracy, f1 = evaluate_model(model, hsi_cube, labels, cfg)
        print(f'Test Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}')

    elif args.mode == 'visualize':
        # Load trained model
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH))
        model.eval()
        # Generate visualizations
        plot_classification_map(model, hsi_cube, labels, cfg)
        plot_confusion_matrix(model, hsi_cube, labels, cfg)
        print(f'Visualizations saved to {cfg.VISUALIZATION_DIR}')

    else:
        raise ValueError(f'Unknown mode: {args.mode}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TGDHTL HSI Classification')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize'],
                        help='Execution mode: train, test, or visualize')
    args = parser.parse_args()
    main(args)
