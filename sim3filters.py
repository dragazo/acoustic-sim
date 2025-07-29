import numpy as np
import torch
import librosa

from abc import ABC, abstractmethod
from typing import Optional

import filter
import filter2
import dataloader
import mfcc
import mfcc_vae_8 as vae
from sim3util import TFLiteModel, energy_chunks   

# Helper to select cluster filter implementation
def make_cluster_filter(max_clusters, max_weight, embedding_size, filter_thresh, distance_metric = 'euclidean'):
    """
    Factory function to create a cluster filter based on the specified method.
    """
    if CLUSTER_METHOD == 'filter2':
        return filter2.ClusterFilter2(max_clusters, max_weight, embedding_size, filter_thresh, distance_metric)
    else:
        return filter.ClusterFilter(max_clusters, max_weight, embedding_size, filter_thresh, distance_metric)

# Default cluster filter method; overridden in main based on args.cluster_method
CLUSTER_METHOD = 'filter'

class FilterMethod(ABC):
    """
    Abstract Base Class for any sample retention method.
    It standardizes the process of initializing a method and using it to make a decision.
    """
    def __init__(self, **kwargs):
        """Initializes the method with specific parameters."""
        pass

    @abstractmethod
    def should_retain(self, audio_clip: np.ndarray) -> bool:
        """
        Processes a single audio clip and returns True if it should be retained, False otherwise.
        This is the core method that every new algorithm must implement.
        """
        pass


class VAEFilter(FilterMethod):
    """
    A filter method that uses a Variational Autoencoder (VAE) to determine whether to retain an audio sample.
    It encodes the audio clip and checks if the encoded representation is within a certain threshold.
    """
    def __init__(self, max_clusters: int, max_weight: float, embedding_size: int, filter_thresh: float, quantized: bool, vote_thresh: float = 0.0, radius: int = 16, chunks: int = 4):
        super().__init__()

        self.vote_thresh = vote_thresh
        self.radius = radius
        self.chunks = chunks

        if quantized:
            device = 'cpu'
            print(f'using tflite quantized model on device "{device}"\n')

            self.encoder = TFLiteModel(model_path = 'model.tflite')
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.device = 'mps' if torch.backends.mps.is_available() else self.device

            print(f'using standard model on device "{self.device}"\n')

            self.encoder = vae.Encoder(embedding_size = embedding_size).to(self.device)
            self.encoder.load_state_dict(torch.load('mfcc-8-untested-4/encoder-F16-A0.5-E256-L22.pt', weights_only = True, map_location = self.device))
            self.encoder.eval()

        # choose cluster filter implementation
        self.filter = make_cluster_filter(max_clusters, max_weight, embedding_size, filter_thresh)

    def should_retain(self, audio_clip: np.ndarray) -> bool:
        """
        Processes the input audio segment through the filter and returns whether the mean of the votes exceeds the defined threshold.
        """

        votes = []

        chunk_size = dataloader.UNIFORM_SAMPLE_RATE * dataloader.SAMPLE_DURATION_SECS
        for seg in energy_chunks(audio_clip, size = chunk_size, count = self.chunks, radius = self.radius):
            with torch.no_grad():
                spectrogram = mfcc.mfcc_spectrogram_for_learning(seg, dataloader.UNIFORM_SAMPLE_RATE)
                mean, _ = self.encoder.forward(torch.tensor(spectrogram[np.newaxis,:], dtype = torch.float).to(self.device))
                votes.append(self.filter.insert(mean.cpu().numpy().squeeze()))

        return np.mean(votes) > self.vote_thresh

class SpectralEncoder:
    def __init__(self, *, fft_size: Optional[int] = None, sample_rate: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size if fft_size is not None else int(30 / 1000 * sample_rate)
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.abs(np.mean(mfcc.spectrogram(x, fft_size = self.fft_size, sample_rate = self.sample_rate), axis = 0))

class SpectralFilter(FilterMethod):
    """
    A filter method that uses spectral features to determine whether to retain an audio sample.
    It computes the mean and variance of the spectrogram and checks if they are within certain thresholds.
    """
    def __init__(self, max_clusters: int, max_weight: float, filter_thresh: float, clip_len: int):
        super().__init__()
        self.encoder = SpectralEncoder(sample_rate = dataloader.UNIFORM_SAMPLE_RATE)
        embedding_size = len(self.encoder.forward(np.zeros((clip_len,))))
        # choose cluster filter implementation
        self.filter = make_cluster_filter(max_clusters, max_weight, embedding_size, filter_thresh)

    def should_retain(self, audio_clip: np.ndarray) -> bool:
        """
        Processes the input audio segment and returns whether it should be retained based on spectral features.
        """
        return self.filter.insert(self.encoder.forward(audio_clip))

class RMSZCEncoder:
    def __init__(self, *, sample_rate: int):
        self.sample_rate = sample_rate
    def forward(self, x: np.ndarray) -> np.ndarray:
        raw = np.array([np.mean(librosa.feature.rms(y = x)), np.mean(librosa.feature.zero_crossing_rate(x))])
        return (raw - np.array([0.12521662, 0.03039862])) / np.array([0.14202671, 0.05519523])

class RMSZCFilter(FilterMethod):
    """
    A filter method that uses RMS and zero-crossing rate to determine whether to retain an audio sample.
    It computes the RMS and zero-crossing rate of the audio clip and checks if they are within certain thresholds.
    """
    def __init__(self, max_clusters: int, max_weight: float, filter_thresh: float, clip_len: int):
        super().__init__()

        self.encoder = RMSZCEncoder(sample_rate = dataloader.UNIFORM_SAMPLE_RATE)
        embedding_size = len(self.encoder.forward(np.zeros((clip_len,))))
        # choose cluster filter implementation
        self.filter = make_cluster_filter(max_clusters, max_weight, embedding_size, filter_thresh)

    def should_retain(self, audio_clip: np.ndarray) -> bool:
        """
        Processes the input audio segment and returns whether it should be retained based on RMS and zero-crossing rate.
        """
        return self.filter.insert(self.encoder.forward(audio_clip))

class CLAPEncoder:
    def __init__(self):
        from transformers import ClapModel, ClapProcessor

        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(0)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    def forward(self, x: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Forward pass through the CLAP model.
        """
        inputs = self.processor(audios=[x], return_tensors="pt", sampling_rate=sample_rate, padding=True).to(0)
        audio_embed = self.model.get_audio_features(**inputs)
        audio_embed = audio_embed.cpu().detach().numpy()
        # Flatten to 1D (512,) to match other encoders
        return audio_embed.squeeze()

class CLAPFilter(FilterMethod):
    """
    A filter method that uses the CLAP model to determine whether to retain an audio sample.
    """
    def __init__(self, max_clusters: int, max_weight: float, filter_thresh: float, clip_len: int, sample_rate: int):
        super().__init__()
        self.encoder = CLAPEncoder()
        embedding_size = len(self.encoder.forward(np.zeros((clip_len,)), sample_rate))
        self.sample_rate = sample_rate
        # choose cluster filter implementation
        self.filter = make_cluster_filter(max_clusters, max_weight, embedding_size, filter_thresh, distance_metric='cosine')

    def should_retain(self, audio_clip: np.ndarray) -> bool:
        """
        Processes the input audio segment and returns whether it should be retained based on CLAP features.
        """
        return self.filter.insert(self.encoder.forward(audio_clip, sample_rate=self.sample_rate))
