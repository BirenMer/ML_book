import pickle
from rnn import RNN
from data_reader import DataReader

def train_rnn(training_file_path, seq_length=25, hidden_size=100, learning_rate=1e-1):
    # Read text from the "input.txt" file
    data_reader = DataReader(f"{training_file_path}", seq_length)
    
    rnn = RNN(hidden_size=hidden_size, vocab_size=data_reader.vocab_size, seq_length=seq_length, learning_rate=learning_rate)
    
    # Train the RNN
    rnn.train(data_reader)

    # Close the data reader file before saving
    data_reader.close()
    
    # Save the trained model and data reader
    with open("rnn_model.pkl", "wb") as f:
        pickle.dump((rnn), f)

    return rnn,data_reader
