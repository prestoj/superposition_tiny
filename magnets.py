import torch
from torch.utils.data import DataLoader
from load_data_pt2 import HiddenStateDataset
from transformers import AutoTokenizer
import torch.optim as optim

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # Load the saved dataset
    dataset_path = "fourth_hidden_state_dataset.pt"
    dataset = torch.load(dataset_path)
    print(len(dataset))

    # Create a DataLoader
    batch_size = 74112 // 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(len(dataloader))

    num_vectors = 64 * 256
    vector_dim = 64
    vectors_tensor = torch.randn(num_vectors, vector_dim).to(device)
    vectors_tensor = vectors_tensor / torch.norm(vectors_tensor, dim=1, keepdim=True)
    vectors_tensor.requires_grad = True

    lr = 1e-2 * 64 / num_vectors
    optimizer = optim.Adam([vectors_tensor], lr=lr)

    # Iterate over the DataLoader
    i_step = 0
    for epoch in range(100):
        for batch in dataloader:
            hidden_layers, raw_tokens = batch
            # raw_tokens = torch.stack(raw_tokens).transpose(0, 1)
            samples = hidden_layers.view(-1, 64)
            samples = samples.to(torch.float32).to(device)
            samples = samples / torch.norm(samples, dim=1, keepdim=True)

            # normalize the vectors
            vectors_tensor.data = vectors_tensor.data / torch.norm(vectors_tensor.data, dim=1, keepdim=True)

            sample_distances = torch.cdist(vectors_tensor, samples, p=2) # (num_vectors, num_samples)
            vector_distances = torch.cdist(vectors_tensor, vectors_tensor, p=2) # (num_vectors, num_vectors)

            sample_distance_loss = torch.min(sample_distances, dim=0).values.mean()
            vector_distance_loss = vector_distances.mean()
            # only look at closest 5 vectors
            # vector_distance_loss = torch.topk(vector_distances, 5, dim=0, largest=False).values.mean()

            loss = sample_distance_loss  - 1 * vector_distance_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if i_step % 100 == 0:
            #     print(f"Step {i_step}, Loss: {loss.item()}, Sample Distance Loss: {sample_distance_loss.item()}, Vector Distance Loss: {vector_distance_loss.item()}")
            #     # print(f"Step {i_step}, Loss: {loss.item()}, Sample Distance Loss: {sample_distance_loss.item()}")

            i_step += 1
        print(f"Epoch {epoch}, Loss: {loss.item()}, Sample Distance Loss: {sample_distance_loss.item()}, Vector Distance Loss: {vector_distance_loss.item()}")

    # Save the vectors
    vectors_path = f"vectors_{num_vectors}_{vector_dim}.pt"
    torch.save(vectors_tensor, vectors_path)


