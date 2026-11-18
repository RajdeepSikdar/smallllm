class SimpleTokenizer:

    def __init__(self, text):
        self.words = sorted(set(text.split()))

        self.word_to_id = {
            word: i for i, word in enumerate(self.words)
        }

        self.id_to_word = {
            i: word for word, i in self.word_to_id.items()
        }

    def encode(self, text):
        return [self.word_to_id[word] for word in text.split()]

    def decode(self, tokens):
        return " ".join(self.id_to_word[token] for token in tokens)


text = "hello how are you hello world"

tokenizer = SimpleTokenizer(text)

encoded = tokenizer.encode("hello world")
decoded = tokenizer.decode(encoded)

print("Vocabulary:", tokenizer.words)
print("Encoded:", encoded)
print("Decoded:", decoded)