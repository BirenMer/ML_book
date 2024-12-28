import os
import pickle
import numpy as np
from run_LSTM import ApplyMyLSTM, RunMyLSTM
from LSTM import LSTM
from layers.dense_layer import DenseLayer
from optimizers.optimizerSGD import  OptimizerSGD
from optimizers.optimizerSGDLSTM import OptimizerSGDLSTM
from prediction_function import generate_text
from model_utils import save_model_to_pkl
from model_utils import load_model_from_pkl

def prepare_text_data(text, seq_length=100):
    chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    text_as_int = np.array([char_to_idx[c] for c in text])
    X, Y = [], []
    for i in range(len(text_as_int) - seq_length):
        X.append(text_as_int[i:i + seq_length])
        Y.append(text_as_int[i + seq_length])
    return np.array(X), np.array(Y).reshape(-1, 1), char_to_idx, idx_to_char


def main(file_path, seq_length=100, n_neurons=256, n_epoch=50, batch_size=64, model_file="lstm_model.pkl"):
    # Load and preprocess the text file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()

    print(f"Text loaded. Total characters: {len(text)}")
    print(f"Unique characters: {len(set(text))}")

    X, Y, char_to_idx, idx_to_char = prepare_text_data(text, seq_length)

    # Check for existing model file
    if os.path.exists(model_file):
        print(f"Model file '{model_file}' found.")
        print("Options:")
        print("1. Load the pre-trained model and skip training.")
        print("2. Load the pre-trained model and continue training.")
        print("3. Train a new model.")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            # Load model and skip training
            lstm, dense_layers, char_to_idx, idx_to_char = load_model_from_pkl(model_file)
            print("Loaded pre-trained model. Skipping training.")
        elif choice == "2":
            # Load model and continue training
            lstm, dense_layers, char_to_idx, idx_to_char = load_model_from_pkl(model_file)
            print("Loaded pre-trained model. Continuing training.")
            RunMyLSTM(X, Y, n_epoch=n_epoch, n_neurons=n_neurons, learning_rate=1e-3, momentum=0.9, batch_size=batch_size)
            save_model_to_pkl(lstm, dense_layers, char_to_idx, idx_to_char, model_file)
        elif choice == "3":
            # Train a new model
            print("Training a new model.")
            RunMyLSTM(X, Y, n_epoch=n_epoch, n_neurons=n_neurons, learning_rate=1e-3, momentum=0.9, batch_size=batch_size)
    else:
        print("No pre-trained model found. Training a new model.")
        RunMyLSTM(X, Y, n_epoch=n_epoch, n_neurons=n_neurons, learning_rate=1e-3, momentum=0.9, batch_size=batch_size)

    # Generate text from the model
    seed_text = "call me ishmael"
    generated_text = generate_text(lstm, dense_layers, seed_text, char_to_idx, idx_to_char, length=500)
    print("\nGenerated Text:\n", generated_text)
main("/home/wifee/workspace/learning/ML_book/LSTM_text_prediction/Mobi_dick_book.txt")