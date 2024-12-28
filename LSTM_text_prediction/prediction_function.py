import numpy as np

def generate_text(lstm, dense_layers, seed_text, char_to_idx, idx_to_char, length=500):
    input_sequence = np.array([char_to_idx[c] for c in seed_text]).reshape(-1, 1)
    generated_text = seed_text

    for _ in range(length):
        # Forward pass
        lstm.forward(input_sequence)
        H = np.array(lstm.H[1:]).reshape(1, -1)

        for layer in dense_layers:
            layer.forward(H)
            H = layer.output

        # Use softmax to get probabilities
        exp_scores = np.exp(H - np.max(H))
        probs = exp_scores / np.sum(exp_scores)

        # Sample next character
        next_char_idx = np.random.choice(len(probs[0]), p=probs[0])
        next_char = idx_to_char[next_char_idx]

        generated_text += next_char
        input_sequence = np.append(input_sequence[1:], [[next_char_idx]], axis=0)

    return generated_text
