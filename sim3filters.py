import numpy as np
import torch
import librosa

from abc import ABC, abstractmethod
from typing import Optional, List

import filter
import filter2
import dataloader
import mfcc
import mfcc_vae_8 as vae
from sim3util import TFLiteModel, energy_chunks   

# Helper to select cluster filter implementation
def make_cluster_filter(max_clusters: int, max_weight: float, embedding_size: int, filter_thresh: float, distance_metric: str = 'euclidean', cluster_method: str = 'filter'):
    """
    Factory function to create a cluster filter based on the specified method.
    
    Args:
        max_clusters: Maximum number of clusters
        max_weight: Maximum weight per cluster
        embedding_size: Size of input embeddings
        filter_thresh: Threshold for filtering
        distance_metric: Distance metric to use
        cluster_method: Cluster filter implementation ('filter' or 'filter2')
        
    Returns:
        Configured cluster filter instance
    """
    if cluster_method == 'filter2':
        return filter2.ClusterFilter2(max_clusters, max_weight, embedding_size, filter_thresh, distance_metric)
    elif cluster_method == 'filter':
        return filter.ClusterFilter(max_clusters, max_weight, embedding_size, filter_thresh, distance_metric)
    else:
        raise ValueError(f"Unknown cluster method: {cluster_method}")

class AudioEncoder(ABC):
    """
    Abstract base class for audio encoders that convert audio clips to feature vectors.
    """
    
    @abstractmethod
    def forward(self, audio_clip: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Encode an audio clip to a feature vector.
        
        Args:
            audio_clip: Raw audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @abstractmethod
    def get_embedding_size(self, clip_len: int, sample_rate: int) -> int:
        """
        Get the size of the embedding vector for a given clip length.
        
        Args:
            clip_len: Length of audio clip in samples
            sample_rate: Sample rate of the audio
            
        Returns:
            Size of the embedding vector
        """
        pass

class FilterBackend(ABC):
    """
    Abstract base class for filtering backends that decide whether to retain samples.
    """
    
    @abstractmethod
    def should_retain(self, embedding: np.ndarray) -> bool:
        """
        Decide whether to retain a sample based on its embedding.
        
        Args:
            embedding: Feature vector from encoder
            
        Returns:
            True if sample should be retained, False otherwise
        """
        pass

class FilterMethod(ABC):
    """
    Abstract base class for complete filter methods that combine encoders and backends.
    """
    
    def __init__(self, encoder: AudioEncoder, backend: FilterBackend):
        """
        Initialize with an encoder and filtering backend.
        
        Args:
            encoder: Audio encoder to generate features
            backend: Filtering backend to make retention decisions
        """
        self.encoder = encoder
        self.backend = backend

    def should_retain(self, audio_clip: np.ndarray, sample_rate: int = None) -> bool:
        """
        Process an audio clip and decide whether to retain it.
        
        Args:
            audio_clip: Raw audio data
            sample_rate: Sample rate (uses dataloader default if None)
            
        Returns:
            True if sample should be retained, False otherwise
        """
        if sample_rate is None:
            sample_rate = dataloader.UNIFORM_SAMPLE_RATE
        
        embedding = self.encoder.forward(audio_clip, sample_rate)
        return self.backend.should_retain(embedding)

# ========== ENCODERS ==========

class VAEEncoder(AudioEncoder):
    """
    VAE-based audio encoder using pre-trained models.
    """
    
    def __init__(self, embedding_size: int, quantized: bool = False, chunks: int = 4, radius: int = 16):
        """
        Initialize VAE encoder.
        
        Args:
            embedding_size: Size of the embedding vector
            quantized: Whether to use TFLite quantized model
            chunks: Number of chunks to process
            radius: Radius for energy chunk selection
        """
        self.embedding_size = embedding_size
        self.chunks = chunks
        self.radius = radius

        if quantized:
            self.device = 'cpu'
            print(f'using tflite quantized model on device "{self.device}"\n')
            self.encoder = TFLiteModel(model_path='model.tflite')
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.device = 'mps' if torch.backends.mps.is_available() else self.device
            print(f'using standard model on device "{self.device}"\n')

            self.encoder = vae.Encoder(embedding_size=embedding_size).to(self.device)
            self.encoder.load_state_dict(torch.load('mfcc-8-untested-4/encoder-F16-A0.5-E256-L22.pt', weights_only=True, map_location=self.device))
            self.encoder.eval()

    def forward(self, audio_clip: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Encode audio clip using VAE.
        """
        embeddings = []
        chunk_size = sample_rate * dataloader.SAMPLE_DURATION_SECS
        
        for seg in energy_chunks(audio_clip, size=chunk_size, count=self.chunks, radius=self.radius):
            with torch.no_grad():
                spectrogram = mfcc.mfcc_spectrogram_for_learning(seg, sample_rate)
                mean, _ = self.encoder.forward(torch.tensor(spectrogram[np.newaxis,:], dtype=torch.float).to(self.device))
                embeddings.append(mean.cpu().numpy().squeeze())
        
        return np.mean(embeddings, axis=0)

    def get_embedding_size(self, clip_len: int, sample_rate: int) -> int:
        """Get embedding size for VAE encoder."""
        return self.embedding_size

class MFCCEncoder(AudioEncoder):
    """
    MFCC-based audio encoder.
    """
    
    def __init__(self, n_mfcc: int = 13):
        """
        Initialize MFCC encoder.
        
        Args:
            n_mfcc: Number of MFCC coefficients
        """
        self.n_mfcc = n_mfcc

    def forward(self, audio_clip: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Encode audio clip using MFCCs.
        """
        mfccs = librosa.feature.mfcc(y=audio_clip, sr=sample_rate, n_mfcc=self.n_mfcc)
        return np.mean(mfccs, axis=1)

    def get_embedding_size(self, clip_len: int, sample_rate: int) -> int:
        """Get embedding size for MFCC encoder."""
        return self.n_mfcc

class SpectralEncoder(AudioEncoder):
    """
    Spectral features encoder.
    """
    
    def __init__(self, fft_size: Optional[int] = None):
        """
        Initialize spectral encoder.
        
        Args:
            fft_size: FFT size (computed from sample rate if None)
        """
        self.fft_size = fft_size

    def forward(self, audio_clip: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Encode audio clip using spectral features.
        """
        fft_size = self.fft_size if self.fft_size is not None else int(30 / 1000 * sample_rate)
        return np.abs(np.mean(mfcc.spectrogram(audio_clip, fft_size=fft_size, sample_rate=sample_rate), axis=0))

    def get_embedding_size(self, clip_len: int, sample_rate: int) -> int:
        """Get embedding size for spectral encoder."""
        fft_size = self.fft_size if self.fft_size is not None else int(30 / 1000 * sample_rate)
        return fft_size // 2 + 1

class RMSZCEncoder(AudioEncoder):
    """
    RMS and Zero-Crossing Rate encoder.
    """
    
    def forward(self, audio_clip: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Encode audio clip using RMS and zero-crossing rate.
        """
        raw = np.array([
            np.mean(librosa.feature.rms(y=audio_clip)), 
            np.mean(librosa.feature.zero_crossing_rate(audio_clip))
        ])
        return (raw - np.array([0.12521662, 0.03039862])) / np.array([0.14202671, 0.05519523])

    def get_embedding_size(self, clip_len: int, sample_rate: int) -> int:
        """Get embedding size for RMS/ZC encoder."""
        return 2

class VAEVotingEncoder(AudioEncoder):
    """
    VAE-based encoder that generates multiple embeddings for voting.
    """
    
    def __init__(self, embedding_size: int, quantized: bool = False, chunks: int = 4, radius: int = 16):
        """
        Initialize VAE voting encoder.
        
        Args:
            embedding_size: Size of the embedding vector
            quantized: Whether to use TFLite quantized model
            chunks: Number of chunks to process
            radius: Radius for energy chunk selection
        """
        self.embedding_size = embedding_size
        self.chunks = chunks
        self.radius = radius

        if quantized:
            self.device = 'cpu'
            print(f'using tflite quantized model on device "{self.device}"\n')
            self.encoder = TFLiteModel(model_path='model.tflite')
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.device = 'mps' if torch.backends.mps.is_available() else self.device
            print(f'using standard model on device "{self.device}"\n')

            self.encoder = vae.Encoder(embedding_size=embedding_size).to(self.device)
            self.encoder.load_state_dict(torch.load('mfcc-8-untested-4/encoder-F16-A0.5-E256-L22.pt', weights_only=True, map_location=self.device))
            self.encoder.eval()

    def forward(self, audio_clip: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """
        Encode audio clip using VAE, returning multiple embeddings for voting.
        """
        embeddings = []
        chunk_size = sample_rate * dataloader.SAMPLE_DURATION_SECS
        
        for seg in energy_chunks(audio_clip, size=chunk_size, count=self.chunks, radius=self.radius):
            with torch.no_grad():
                spectrogram = mfcc.mfcc_spectrogram_for_learning(seg, sample_rate)
                mean, _ = self.encoder.forward(torch.tensor(spectrogram[np.newaxis,:], dtype=torch.float).to(self.device))
                embeddings.append(mean.cpu().numpy().squeeze())
        
        return embeddings

    def get_embedding_size(self, clip_len: int, sample_rate: int) -> int:
        """Get embedding size for VAE encoder."""
        return self.embedding_size

class CLAPEncoder(AudioEncoder):
    """
    CLAP model encoder.
    """
    
    def __init__(self):
        """Initialize CLAP encoder."""
        from transformers import ClapModel, ClapProcessor
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(0)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    def forward(self, audio_clip: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Encode audio clip using CLAP.
        """
        inputs = self.processor(audios=[audio_clip], return_tensors="pt", sampling_rate=sample_rate, padding=True).to(0)
        audio_embed = self.model.get_audio_features(**inputs)
        return audio_embed.cpu().detach().numpy().squeeze()

    def get_embedding_size(self, clip_len: int, sample_rate: int) -> int:
        """Get embedding size for CLAP encoder."""
        return 512  # CLAP embedding size

# ========== FILTER BACKENDS ==========

class ClusterFilterBackend(FilterBackend):
    """
    Clustering-based filter backend.
    """
    
    def __init__(self, max_clusters: int, max_weight: float, embedding_size: int, filter_thresh: float, distance_metric: str = 'euclidean', cluster_method: str = 'filter'):
        """
        Initialize cluster filter backend.
        
        Args:
            max_clusters: Maximum number of clusters
            max_weight: Maximum weight per cluster
            embedding_size: Size of input embeddings
            filter_thresh: Threshold for filtering
            distance_metric: Distance metric to use
            cluster_method: Cluster filter implementation ('filter' or 'filter2')
        """
        self.filter = make_cluster_filter(max_clusters, max_weight, embedding_size, filter_thresh, distance_metric, cluster_method)

    def should_retain(self, embedding: np.ndarray) -> bool:
        """
        Decide retention based on clustering.
        """
        return self.filter.insert(embedding)

class VotingFilterBackend(FilterBackend):
    """
    Voting-based filter backend that uses multiple embeddings.
    """
    
    def __init__(self, base_backend: FilterBackend, vote_thresh: float = 0.0):
        """
        Initialize voting filter backend.
        
        Args:
            base_backend: Backend to use for individual votes
            vote_thresh: Threshold for average vote
        """
        self.base_backend = base_backend
        self.vote_thresh = vote_thresh

    def should_retain(self, embeddings: List[np.ndarray]) -> bool:
        """
        Decide retention based on voting across multiple embeddings.
        """
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        
        votes = [self.base_backend.should_retain(emb) for emb in embeddings]
        return np.mean(votes) > self.vote_thresh

# ========== CONVENIENCE FUNCTIONS ==========

def create_filter_method(encoder_type: str, backend_type: str = 'cluster', **kwargs) -> FilterMethod:
    """
    Factory function to create filter methods from encoder and backend types.
    
    Args:
        encoder_type: Type of encoder ('vae', 'mfcc', 'spectral', 'rmszc', 'clap')
        backend_type: Type of backend ('cluster', 'voting')
        **kwargs: Additional arguments for encoder and backend
        
    Returns:
        Configured FilterMethod instance
    """
    # Create encoder
    if encoder_type == 'vae':
        if backend_type == 'voting':
            encoder = VAEVotingEncoder(**{k: v for k, v in kwargs.items() if k in ['embedding_size', 'quantized', 'chunks', 'radius']})
        else:
            encoder = VAEEncoder(**{k: v for k, v in kwargs.items() if k in ['embedding_size', 'quantized', 'chunks', 'radius']})
    elif encoder_type == 'mfcc':
        encoder = MFCCEncoder(**{k: v for k, v in kwargs.items() if k in ['n_mfcc']})
    elif encoder_type == 'spectral':
        encoder = SpectralEncoder(**{k: v for k, v in kwargs.items() if k in ['fft_size']})
    elif encoder_type == 'rmszc':
        encoder = RMSZCEncoder()
    elif encoder_type == 'clap':
        encoder = CLAPEncoder()
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Create backend
    if backend_type == 'cluster':
        clip_len = kwargs.get('clip_len', dataloader.UNIFORM_SAMPLE_RATE * dataloader.SAMPLE_DURATION_SECS)
        sample_rate = kwargs.get('sample_rate', dataloader.UNIFORM_SAMPLE_RATE)
        embedding_size = encoder.get_embedding_size(clip_len, sample_rate)
        
        backend = ClusterFilterBackend(
            max_clusters=kwargs['max_clusters'],
            max_weight=kwargs['max_weight'],
            embedding_size=embedding_size,
            filter_thresh=kwargs['filter_thresh'],
            distance_metric=kwargs.get('distance_metric', 'euclidean'),
            cluster_method=kwargs.get('cluster_method', 'filter')
        )
    elif backend_type == 'voting':
        clip_len = kwargs.get('clip_len', dataloader.UNIFORM_SAMPLE_RATE * dataloader.SAMPLE_DURATION_SECS)
        sample_rate = kwargs.get('sample_rate', dataloader.UNIFORM_SAMPLE_RATE)
        embedding_size = encoder.get_embedding_size(clip_len, sample_rate)
        
        base_backend = ClusterFilterBackend(
            max_clusters=kwargs['max_clusters'],
            max_weight=kwargs['max_weight'],
            embedding_size=embedding_size,
            filter_thresh=kwargs['filter_thresh'],
            distance_metric=kwargs.get('distance_metric', 'euclidean'),
            cluster_method=kwargs.get('cluster_method', 'filter')
        )
        backend = VotingFilterBackend(base_backend, kwargs.get('vote_thresh', 0.0))
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    return FilterMethod(encoder, backend)