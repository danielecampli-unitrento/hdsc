import os
import numpy as np
import xarray as xr
import warnings
import torchhd
import torch
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def embed(data, in_features, out_features, embed_type="random", vsa="BSC"):
    mapping = {
        "random": torchhd.embeddings.Random,
        "level": torchhd.embeddings.Level,
        "density": torchhd.embeddings.Density
    }
    if embed_type not in mapping:
        raise ValueError(f"Unknown embedding type: {embed_type}")
    embedding = mapping[embed_type](in_features, out_features, vsa=vsa)
    return embedding(data)

if __name__ == "__main__":
    # Ignore annoying urllib3 warnings
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

    input_folder = 'data/ncdf'
    output_folder = 'data/ncdf'

    # Runtime configuration
    os.makedirs(output_folder, exist_ok=True)
    
    dataset = xr.load_dataarray(os.path.join(input_folder, "nrm.orx.dataset.nc"))

    no_radii = dataset.radius.size
    no_frequencies = dataset.frequency.size
    embedding_size = 32768
    random_embedding_size = 1024
    vsa_encoding = "BSC"

    frequency_embeddings = embed(
        torch.randn(no_frequencies, random_embedding_size),
        in_features=random_embedding_size,
        out_features=embedding_size,
        embed_type="density",
        vsa=vsa_encoding)

    radius_embeddings = embed(
        torch.randn(no_radii, random_embedding_size),
        in_features=random_embedding_size,
        out_features=embedding_size,
        embed_type="density",
        vsa=vsa_encoding)

    tensor_dataset =  torch.tensor(dataset.values).permute(1, 0, 2)

    frequencies = tensor_dataset.shape[0]
    radii = tensor_dataset.shape[1]
    signal_embeddings_by_radius = []
    for frequency in np.arange(frequencies):
        for radius in np.arange(radii):
            signal = tensor_dataset[frequency, radius, :]
            print(frequency, radius, signal)
            signal_embedding = embed(signal, dataset.time.size, embedding_size, "density", vsa_encoding)
            signal_embeddings_by_radius.append(signal_embedding)
        kbe = torchhd.hash_table(radius_embeddings, signal_embeddings_by_radius)
        print(f"Embedding frequency {frequency + 1} of {dataset.frequency.size}")
    kbe = torchhd.hash_table(frequency_embeddings, signal_embeddings_by_radius)
    print(kbe)