
import torch

from tokenizer import SimpleTokenizer
from model import GPTLanguageModel


device = torch.device("cpu")

# Load checkpoint
checkpoint = torch.load(
    "my_llm.pth",
    map_location=device,
    weights_only=True
)

# Restore tokenizer
tokenizer = SimpleTokenizer(
    " ".join(checkpoint["words"])
)

config = checkpoint["config"]

# Rebuild model
model = GPTLanguageModel(**config).to(device)

model.load_state_dict(checkpoint["model_state"])

model.eval()

print("My LLM is ready!")
print("Type 'exit' to stop.\n")

while True:

    prompt = input("You: ").strip()

    if prompt.lower() == "exit":
        break

    try:

        input_tokens = tokenizer.encode(prompt)

        x = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=device
        )

        output = model.generate(
            x,
            max_new_tokens=20,
            temperature=0.8
        )

        generated_tokens = output[0].tolist()

        response = tokenizer.decode(generated_tokens)

        print("\nAI:", response)
        print()

    except KeyError:
        print("Some words are not in the training vocabulary.")
