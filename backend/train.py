
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import SimpleTokenizer
from dataset import TextDataset
from model import GPTLanguageModel


# 1. Training data
text = """
hello how are you
i am learning artificial intelligence
python is a programming language
machine learning is interesting
deep learning uses neural networks
transformers understand relationships in text
i love building software
i want to create my own language model
artificial intelligence is changing technology
programming requires practice and patience
how are you today
i am doing well
"""

# Repeat data to create more training examples
text = text * 100

# 2. Configuration
BLOCK_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

device = torch.device("cpu")

# 3. Tokenizer
tokenizer = SimpleTokenizer(text)

vocab_size = len(tokenizer.words)

print("Vocabulary size:", vocab_size)

# 4. Dataset
dataset = TextDataset(
    text,
    tokenizer,
    block_size=BLOCK_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 5. Model
model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    embed_dim=128,
    heads=4,
    layers=2
).to(device)

# 6. Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_function = nn.CrossEntropyLoss()

# 7. Training loop
for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = loss_function(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Loss: {average_loss:.4f}"
    )

# 8. Save trained model
torch.save(
    {
        "model_state": model.state_dict(),
        "words": tokenizer.words,
        "config": {
            "vocab_size": vocab_size,
            "block_size": BLOCK_SIZE,
            "embed_dim": 128,
            "heads": 4,
            "layers": 2
        }
    },
    "my_llm.pth"
)

print("\nTraining completed!")
print("Model saved as my_llm.pth")
