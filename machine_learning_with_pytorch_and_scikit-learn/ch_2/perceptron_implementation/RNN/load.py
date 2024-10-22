import pickle

def load_rnn(model_save_path="rnn_model.pkl"):
    with open(model_save_path, 'rb') as f:
        rnn = pickle.load(f)
    return rnn
