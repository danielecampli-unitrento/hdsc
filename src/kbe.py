import os
import numpy as np
import xarray as xr
import warnings
import torchhd
import torch
from tqdm import tqdm
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
    signal_embeddings = []

    total_iterations = frequencies * radii

    with tqdm(total=total_iterations, desc="Embedding signals", unit="pair") as pbar:
        for frequency in range(frequencies):
            signal_embeddings_by_radius = []
            for radius in range(radii):
                signal = tensor_dataset[frequency, radius, :]
                signal_embedding = embed(signal, dataset.time.size, embedding_size, "density", vsa_encoding)
                signal_embeddings_by_radius.append(signal_embedding)
                
                # Update progress
                pbar.update(1)
                # Optional: show which (freq, radius) pair is processing
                pbar.set_postfix(freq=frequency+1, radius=radius*10+10)
            
            signal_embeddings.append(signal_embeddings_by_radius)

    embeddings = xr.DataArray(
        signal_embeddings,
        dims=('frequency', 'radius', 'dimension'),
        coords={
            'frequency': np.arange(dataset.frequency.size) + 1,
            'radius': np.arange(dataset.radius.size) * 10 + 10,
            'dimension': np.arange(embedding_size) + 1
        }
    )
    embeddings.to_netcdf(os.path.join(output_folder, "bsc.embeddings2.nc"))