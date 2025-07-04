
# --- 1. Vocabulary and Tokenizer ---
# We'll create a simple character-level vocabulary for SMILES strings.

class SMILES_Tokenizer:
    """
    A simple tokenizer for SMILES strings. It creates a vocabulary from a list
    of SMILES and converts strings to sequences of integers and back.
    """
    def __init__(self):
        # Special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<sos>" # Start of Sequence
        self.eos_token = "<eos>" # End of Sequence
        self.unk_token = "<unk>" # Unknown token

        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0

    def fit(self, smiles_list):
        """
        Creates the vocabulary from a list of SMILES strings.
        """
        all_chars = set()
        for smiles in smiles_list:
            all_chars.update(list(smiles))

        # Sort for reproducibility
        sorted_chars = sorted(list(all_chars))

        # Add special tokens first
        special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        for i, token in enumerate(special_tokens):
            self.char_to_int[token] = i
            self.int_to_char[i] = token

        # Add character tokens
        for i, char in enumerate(sorted_chars, start=len(special_tokens)):
            self.char_to_int[char] = i
            self.int_to_char[i] = char

        self.vocab_size = len(self.char_to_int)
        print(f"Vocabulary created. Size: {self.vocab_size}")

    def encode(self, smiles):
        """
        Converts a SMILES string to a list of integers, adding SOS and EOS tokens.
        """
        encoded = [self.char_to_int[self.sos_token]]
        for char in smiles:
            encoded.append(self.char_to_int.get(char, self.char_to_int[self.unk_token]))
        encoded.append(self.char_to_int[self.eos_token])
        return encoded

    def decode(self, int_sequence):
        """
        Converts a list of integers back to a SMILES string.
        Stops at the first EOS token.
        """
        chars = []
        for i in int_sequence:
            char = self.int_to_char.get(i)
            if char == self.eos_token:
                break
            if char not in [self.sos_token, self.pad_token]:
                chars.append(char)
        return "".join(chars)