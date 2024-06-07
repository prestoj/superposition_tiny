import torch
from torch.utils.data import DataLoader
from load_data_pt2 import HiddenStateDataset
from transformers import AutoTokenizer
import torch.optim as optim

def allocate_seats(votes, num_seats):
    # Create a list of parties and their vote counts
    parties = list(votes.keys())
    vote_counts = list(votes.values())

    # Initialize the seat allocation for each party to 0
    seats = {party: 0 for party in parties}

    # Allocate seats using the D'Hondt method
    for _ in range(num_seats):
        # Find the party with the highest quotient (votes / (seats + 1))
        quotients = [count / (seats[party] + 1) for party, count in zip(parties, vote_counts)]
        winner = parties[quotients.index(max(quotients))]

        # Allocate a seat to the winning party
        seats[winner] += 1

    return seats

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # Load the saved dataset
    dataset_path = "fourth_hidden_state_dataset.pt"
    dataset = torch.load(dataset_path)

    # Create a DataLoader
    batch_size = 74112 // 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_vectors = 64 * 16
    vector_dim = 64
    vectors_tensor = torch.randn(num_vectors, vector_dim).to(device)
    vectors_tensor = vectors_tensor / torch.norm(vectors_tensor, dim=1, keepdim=True)
    vectors_tensor.requires_grad = True

    lr = 1e-3 * 74112 / batch_size
    optimizer = optim.Adam([vectors_tensor], lr=lr)

    # Iterate over the DataLoader
    i_step = 0
    for epoch in range(100):
        alive_vectors_samples = {i: 0 for i in range(num_vectors)}
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
            # vector_distance_loss = vector_distances.mean()
            # only look at closest 5 vectors
            vector_distance_loss = torch.topk(vector_distances, 5, dim=0, largest=False).values.mean()

            loss = sample_distance_loss  - 1 * vector_distance_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i_step += 1

            with torch.no_grad():
                for i_vector in range(num_vectors):
                    closest_indices = (sample_distances.argmin(dim=0) == i_vector).nonzero().squeeze()
                    if closest_indices.numel() > 0:
                        if closest_indices.dim() == 0:
                            closest_indices = closest_indices.unsqueeze(0)
                        closest_indices = closest_indices.tolist()

                        num_samples = len(closest_indices)
                        alive_vectors_samples[i_vector] += num_samples
            
        dead_vectors = [i_vector for i_vector, num_samples in alive_vectors_samples.items() if num_samples == 0]
        with torch.no_grad():
            # print min median max of non-zero alive_vectors_samples
            alive_vectors_samples_non_zero = {k: v for k, v in alive_vectors_samples.items() if v > 0}
            num_samples = list(alive_vectors_samples_non_zero.values())
            print(f"Epoch {epoch}, Min: {min(num_samples)}, Median: {sorted(num_samples)[len(num_samples) // 2]}, Max: {max(num_samples)}")
            if len(dead_vectors) > 0:
                vectors_allocation = allocate_seats(alive_vectors_samples, len(dead_vectors))
                i_dead_vector = 0
                new_vectors_tensor = vectors_tensor.clone().detach()
                for i_vector, num_allocated in vectors_allocation.items():
                    for _ in range(num_allocated):
                        new_vector = vectors_tensor[i_vector].detach() + (torch.rand(vector_dim).to(device) * 2 - 1) * 1e-3
                        new_vectors_tensor[dead_vectors[i_dead_vector]] = new_vector
                        i_dead_vector += 1
                vectors_tensor = new_vectors_tensor.clone().detach().requires_grad_(True)
                # optimizer_params = optimizer.state_dict()
                optimizer = optim.Adam([vectors_tensor], lr=lr)
                # optimizer.load_state_dict(optimizer_params)

        print(f"Epoch {epoch}, Loss: {loss.item()}, Sample Distance Loss: {sample_distance_loss.item()}, Vector Distance Loss: {vector_distance_loss.item()}, Dead Vectors: {len(dead_vectors)}")
        
    # Save the vectors
    vectors_path = f"vectors_pruned_{num_vectors}_{vector_dim}.pt"
    torch.save(vectors_tensor, vectors_path)


