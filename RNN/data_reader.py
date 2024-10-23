import numpy as np

# Class to read the training data, create vocabulary, and generate indexed batches.
class DataReader:
    def __init__(self, path, seq_length):
        """
        Initializes the DataReader by reading the text from a file or a string,
        creates mappings from characters to indices and vice versa.
        
        Parameters:
        - path: Path to the text file containing the data
        - seq_length: Length of the sequence for the training batches
        """

        # Optional: Uncomment below if you don't want to use any file for text reading
        # and comment the next two lines.
        #self.data = "some really long text to test this. maybe not perfect but should get you going."
        
        # Open and read the input file
        self.fp = open(path, "r")
        self.data = self.fp.read()

        # Find unique characters in the data
        chars = list(set(self.data))
        
        # Create dictionary mappings for each character to an index and vice versa
        self.char_to_ix = {ch: i for (i, ch) in enumerate(chars)}
        self.ix_to_char = {i: ch for (i, ch) in enumerate(chars)}
        
        # Total size of the data
        self.data_size = len(self.data)
        
        # Number of unique characters (vocabulary size)
        self.vocab_size = len(chars)
        
        # Pointer to track the current batch start position
        self.pointer = 0
        
        # Length of each sequence for training
        self.seq_length = seq_length

    def next_batch(self):
        """
        Generates the next batch of input and target sequences.

        Returns:
        - inputs: List of integers representing the current batch of input characters (as indices).
        - targets: List of integers representing the target characters (as indices) for the next time step.
        """
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        
        # Convert characters to indices for inputs and targets
        inputs = [self.char_to_ix[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_ix[ch] for ch in self.data[input_start + 1:input_end + 1]]
        
        # Move the pointer forward by the sequence length
        self.pointer += self.seq_length
        
        # If we reach the end of the data, reset the pointer to zero (loop over the data)
        if self.pointer + self.seq_length + 1 >= self.data_size:
            self.pointer = 0
        
        return inputs, targets

    def just_started(self):
        """
        Checks if the DataReader has just started a new epoch (i.e., if the pointer is at the start of the data).
        Returns:
        - Boolean indicating whether the DataReader just started reading.
        """
        return self.pointer == 0

    def close(self):
        """
        Closes the file that was opened for reading.
        """
        self.fp.close()
