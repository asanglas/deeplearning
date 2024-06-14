
class CharacterTokenizer:
    def __init__(self, text):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.char_to_idx = {ch:i for i,ch in enumerate(self.chars)}
        self.idx_to_char = {i:ch for i,ch in enumerate(self.chars)}
        
    def encode(self, string):
        return [self.char_to_idx[ch] for ch in string]

    def decode(self, array):
        return ''.join([self.idx_to_char[i] for i in array])

    def get_vocab_size(self):
        return self.vocab_size