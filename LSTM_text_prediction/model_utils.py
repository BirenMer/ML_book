import pickle

def save_model_to_pkl(lstm, dense_layers, char_to_idx, idx_to_char, filename="lstm_model.pkl"):
    model_data = {
        "lstm": lstm,
        "dense_layers": dense_layers,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }
    with open(filename, "wb") as file:
        pickle.dump(model_data, file)
    print(f"Model saved to {filename}")

def load_model_from_pkl(filename="lstm_model.pkl"):
    with open(filename, "rb") as file:
        model_data = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model_data["lstm"], model_data["dense_layers"], model_data["char_to_idx"], model_data["idx_to_char"]
