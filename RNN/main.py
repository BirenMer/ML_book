from train import train_rnn
from load import load_rnn

training_file_path="/home/wifee/workspace/learning/ML_book/Mobi_dick_book.txt"

if __name__ == "__main__":
    model,data_reader = train_rnn(training_file_path)
    
    # Save model is done inside train_rnn, but you can load later for prediction (I personally don't recommend doing so as of now it affects the quality of the output provided)
    # rnn = load_rnn("rnn_model.pkl")
    
    # Example of making a prediction
    start_text = "Hello"
    predicted_text = model.predict(data_reader, start_text, n=200)
    print(predicted_text)
