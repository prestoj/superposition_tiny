import torch
from torch.utils.data import DataLoader
from load_data_pt2 import HiddenStateDataset
from transformers import AutoTokenizer
import torch.optim as optim
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Load the saved dataset
dataset_path = "fourth_hidden_state_dataset.pt"
dataset = torch.load(dataset_path)

# Create a DataLoader
# batch_size = 74112 // 1
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

num_vectors = 256 * 1
vector_dim = 64
vectors_tensor = torch.load(f'vectors_{num_vectors}_{vector_dim}.pt')
vectors_tensor = vectors_tensor / torch.norm(vectors_tensor, dim=1, keepdim=True)

def generate_color(num):
    # Generate a random color based on the mapped number
    random.seed(num)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"\033[38;2;{r};{g};{b}m"

reset_color = "\033[0m"  # Reset color code

for batch in dataloader:
    hidden_layers, raw_tokens = batch
    raw_tokens = torch.stack(raw_tokens).transpose(0, 1)
    samples = hidden_layers.view(-1, 64)
    samples = samples.to(torch.float32).to(device)
    samples = samples / torch.norm(samples, dim=1, keepdim=True)

    # normalize the vectors
    vectors_tensor.data = vectors_tensor.data / torch.norm(vectors_tensor.data, dim=1, keepdim=True)

    sample_distances = torch.cdist(vectors_tensor, samples, p=2)  # (num_vectors, num_samples)
    vector_distances = torch.cdist(vectors_tensor, vectors_tensor, p=2)  # (num_vectors, num_vectors)

    sample_distance_loss = torch.min(sample_distances, dim=0).values.mean()

    used_vectors = []
    for i in range(10):
        sentence = tokenizer.decode(raw_tokens[i])
        decoded_tokens = tokenizer.convert_ids_to_tokens(raw_tokens[i])
        mapped_numbers = sample_distances.argmin(dim=0).view(batch_size, 64)[i].tolist() # 64 for the sequence length
        for num in mapped_numbers:
            if num not in used_vectors:
                used_vectors.append(num)

        # Replace "Ġ" with actual spaces in the decoded tokens
        decoded_tokens = [token.replace("Ġ", " ").replace("Ċ", "\n") for token in decoded_tokens]
        
        colored_sentence = "".join([f"{generate_color(num)}{token}[{num}]{reset_color}" for token, num in zip(decoded_tokens, mapped_numbers)])
        print('-'*50)
        print(colored_sentence)
        print('-'*50)
    print(used_vectors)
    break

# dead_vectors = 0
# for i_vector in range(num_vectors):
#     closest_indices = (sample_distances.argmin(dim=0) == i_vector).nonzero().squeeze()
#     if closest_indices.numel() > 0:
#         if closest_indices.dim() == 0:
#             closest_indices = closest_indices.unsqueeze(0)
#         closest_indices = closest_indices.tolist()

#         num_samples = len(closest_indices)
#         print(f"Vector {i_vector} has {num_samples} samples")
#     else:
#         dead_vectors += 1

# print(dead_vectors)


for i_vector in range(num_vectors):
    print(f"Vector {i_vector}")
    print(vectors_tensor[i_vector])