import os
import random
import math
import torch
import dataloader
import numpy as np
import librosa
import tensorflow as tf
from typing import Dict

def qprint(*args, **kwargs):
    QUIET = globals().get('QUIET', True)
    """Prints messages to the console unless QUIET is set to True."""
    if not QUIET:
        print(*args, **kwargs)

def load_sounds(path: str, *, min_length: float = 0, max_length: float = math.inf, mult_length: float = None, max_silence_ratio: float = None, sample_rate: int = None) -> Dict[str, np.ndarray]:
    """Loads audio files from a directory and filters them based on various criteria.

    Args:
        path (str): Directory path.
        min_length (float, optional): Minimum clip length in seconds.
        max_length (float, optional): Maximum clip length in seconds.
        mult_length (float, optional): Required multiple of length in seconds.
        max_silence_ratio (float, optional): Maximum allowed silence ratio.
        sample_rate (int, optional): Audio sample rate to load. Defaults to dataloader.UNIFORM_SAMPLE_RATE.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping class names to their audio clips.
    """
    if sample_rate is None:
        sample_rate = dataloader.UNIFORM_SAMPLE_RATE
    res = { cls: [] for cls in sorted(os.listdir(path)) }

    for cls, entries in res.items():
        for file in sorted(os.listdir(f'{path}/{cls}')):
            if not file.lower().endswith(('.wav', '.flac', '.mp3', '.ogg', '.aiff', '.aif', '.aifc')):
                qprint(f'  omitting {path}/{cls}/{file} -- not a supported audio format')
                continue
            
            clip, sr = librosa.load(f'{path}/{cls}/{file}', sr = sample_rate)

            assert sr == sample_rate, sr
            assert len(clip.shape) == 1, clip.shape

            if mult_length is not None:
                assert len(clip) % (mult_length * sr) == 0, f'{path}/{cls}/{file} not a multiple of {mult_length}s'

            if len(clip) < min_length * sr:
                qprint(f'  omitting {path}/{cls}/{file} -- too short ({len(clip) / sr:0.2f}s < {min_length:0.2f}s)')
                continue
            if len(clip) > max_length * sr:
                qprint(f'  omitting {path}/{cls}/{file} -- too long ({len(clip) / sr:0.2f}s > {max_length:0.2f}s)')
                continue

            if max_silence_ratio is not None and np.mean(clip == 0) > max_silence_ratio:
                qprint(f'  omitting {path}/{cls}/{file} -- too much silence')
                continue

            entries.append(clip)

    return { k: v for k,v in res.items() if len(v) > 0 }

def random_contract(x: np.ndarray, s: int) -> np.ndarray:
    """
    Randomly extracts a segment of length `s` from the input array `x`.
    
    Args:
        x (np.ndarray): The input array from which to extract a segment.
        s (int): The desired length of the segment to extract.

    Returns:
        np.ndarray: A segment of length `s` extracted from `x`. If `x` is shorter than `s`, it will be returned as is.
    """
    if x.shape[0] <= s: return x
    t = random.randrange(x.shape[0] - s)
    return x[t:t+s]

def random_extend(x: np.ndarray, s: int) -> np.ndarray:
    """
    Randomly extends the input array `x` to a length of `s` by padding with zeros if necessary.
    Args:
        x (np.ndarray): The input array to extend.
        s (int): The desired length of the output array.

    Returns:
        np.ndarray: The extended array of length `s`. If `x` is already of length `s` or longer, it is returned unchanged.
    """
    if x.shape[0] >= s: return x
    t = random.randrange(s - x.shape[0])
    res = np.zeros((s,))
    res[t:t+x.shape[0]] = x
    return res

def create_fade(size: int, *, fade_duration: float, sr: int) -> np.ndarray:
    """
    Creates a fade-in and fade-out effect for an audio clip.

    Args:
        size (int): The total size of the fade effect.
        fade_duration (float): The duration of the fade-in and fade-out in seconds.
        sr (float): The sample rate of the audio.

    Returns:
        np.ndarray: An array representing the fade effect, where the first part fades in, the middle is constant, and the last part fades out.
    """
    t = min(round(fade_duration * sr), round(size / 4))
    return np.concatenate([
        np.linspace(0, 1, t),
        np.ones((size - 2 * t,)),
        np.linspace(1, 0, t),
    ])

def energy_peak(x: np.array, *, radius: int) -> int:
    """
    Finds the peak of the energy in the input array `x` using a convolution with a kernel that emphasizes the center and tapers off towards the edges.

    Args:
        x (np.array): The input array representing the audio signal.
        radius (int): The radius of the kernel used for convolution.

    Returns:
        int: The index of the peak energy in the input array `x`.
    """
    kernel = np.concatenate([
        np.linspace(0, 1, radius + 2)[1:-1],
        [1],
        np.linspace(1, 0, radius + 2)[1:-1],
    ])
    return np.argmax(np.convolve(x**2, kernel, mode = 'same'))

def energy_chunks(x: np.array, *, size: int, radius: int, count: int) -> np.array:
    """
    Splits the input array `x` into chunks of a specified size, centered around peaks of energy, while ensuring that the chunks do not overlap.

    Args:
        x (np.array): The input array representing the audio signal.
        size (int): The size of each chunk.
        radius (int): The radius of the kernel used for convolution.
        count (int): The number of chunks to extract.

    Returns:
        np.array: An array of shape (count, size) containing the extracted chunks.
    """

    if count == 0:
        return np.split(x, len(x) // size)

    res = []
    x = np.copy(x)
    half = size // 2
    e = np.concatenate([np.zeros((half,)), x, np.zeros((half,))])

    for _ in range(count):
        p = energy_peak(x, radius=radius)
        res.append(e[p : p + size])
        x[max(0, p - half) : min(len(x), p - half + size)] = 0
    
    return np.array(res)

def chunks(x: np.array, *, size: int, overlaps: int) -> np.array:
    """
    Splits the input array `x` into overlapping chunks of a specified size.

    Args:
        x (np.array): The input array to be split into chunks.
        size (int): The size of each chunk.
        overlaps (int): The number of overlapping elements between consecutive chunks.

    Returns:
        np.array: An array of shape (n, size) containing the chunks, where n is the number of chunks extracted from `x`.
    """

    assert len(x) >= size

    p = 0
    s = round(size / (1 + overlaps))
    res = []

    while p + size <= len(x):
        res.append(x[p:p+size])
        p += s

    return np.array(res)

class TFLiteModel:
    """
    Wrapper for TensorFlow Lite models to allow them to be used in a PyTorch-like manner.
    """

    def __init__(self, *args, **kwargs):
        self.model = tf.lite.Interpreter(*args, **kwargs)
        self.model.allocate_tensors()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TensorFlow Lite model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """

        self.model.set_tensor(self.model.get_input_details()[0]['index'], x)
        self.model.invoke()

        return (torch.tensor(self.model.get_tensor(self.model.get_output_details()[0]['index'])), None)

def kl_divergence_uniform(output_events: Dict[str, int]) -> float:
        """
        Compute the KL divergence of the empirical output event distribution FROM a uniform distribution.
        This measures how much the output distribution deviates from uniformity.
        KL(Q || P) = sum(Q(x) * log(Q(x) / P(x)))

        Args:
            output_events: A dictionary mapping event names to their output counts (Q).

        Returns:
            The KL divergence (float). Returns 0.0 if total output is zero.
        """
        total_output = sum(output_events.values())
        num_events = len(output_events)

        if total_output == 0 or num_events == 0:
            return 0.0

        uniform_prob = 1.0 / num_events
        kl_divergence = 0.0

        for event_count in output_events.values():
            if event_count > 0:
                q_prob = event_count / total_output
                kl_divergence += q_prob * np.log(q_prob / uniform_prob)
        
        return kl_divergence
