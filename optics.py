import torch
from torch.utils.data import DataLoader
from load_data_pt2 import HiddenStateDataset
import torch.optim as optim
from sklearn.cluster import OPTICS
import numpy as np

if __name__ == "__main__":

    # Load the saved dataset
    dataset_path = "fourth_hidden_state_dataset.pt"
    dataset = torch.load(dataset_path)

    # Create a DataLoader
    batch_size = 74112 // 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in dataloader:
        hidden_layers, raw_tokens = batch
        # raw_tokens = torch.stack(raw_tokens).transpose(0, 1)
        samples = hidden_layers.view(-1, 64)
        samples = samples.to(torch.float32)
        # samples = samples / torch.norm(samples, dim=1, keepdim=True)

        samples = samples.numpy()
        samples = samples[:100000]

        clustering = OPTICS(min_samples=2).fit(samples)
        print(clustering.labels_.shape)
        print((clustering.labels_ == -1).mean())
        print(np.unique(clustering.labels_).shape)



