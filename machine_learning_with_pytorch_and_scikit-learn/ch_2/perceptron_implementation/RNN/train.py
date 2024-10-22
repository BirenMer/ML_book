import pickle
from data_reader import DataReader
from rnn import RNN

def train_rnn(training_file_path, seq_length=25, hidden_size=100, learning_rate=1e-1, model_save_path="rnn_model.pkl"):
    # read text from the training file
    data_reader = DataReader(training_file_path, seq_length)
    
    # create the RNN model
    rnn = RNN(
        hidden_size=hidden_size,
        vocab_size=data_reader.vocab_size,
        seq_length=seq_length,
        learning_rate=learning_rate,
    )
    
    # train the model
    rnn.train(data_reader)
    
    # save the model using pickle
    with open(model_save_path, 'wb') as f:
        pickle.dump(rnn, f)
    
    return rnn
