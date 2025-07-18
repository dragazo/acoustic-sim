import argparse
import librosa
import dataloader
import random
import math
import os
import torch
import soundfile
import mfcc_vae_8 as vae
import mfcc
import numpy as np
import filter
import tensorflow as tf

from typing import Dict, Any

# Set QUIET to True to suppress console output
QUIET = False

def qprint(*args, **kwargs):
    """Prints messages to the console unless QUIET is set to True."""
    if not QUIET:
        print(*args, **kwargs)

def load_sounds(path: str, *, min_length: float = 0, max_length: float = math.inf, mult_length: float = None, max_silence_ratio: float = None) -> Dict[str, np.ndarray]:
    """Loads audio files from a directory and filters them based on various criteria.

    Args:
        path (str): _description_
        min_length (float, optional): _description_. Defaults to 0.
        max_length (float, optional): _description_. Defaults to math.inf.
        mult_length (float, optional): _description_. Defaults to None.
        max_silence_ratio (float, optional): _description_. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping class names to their audio clips.
    """
    res = { cls: [] for cls in sorted(os.listdir(path)) }

    for cls, entries in res.items():
        for file in sorted(os.listdir(f'{path}/{cls}')):
            clip, sr = librosa.load(f'{path}/{cls}/{file}', sr = dataloader.UNIFORM_SAMPLE_RATE)

            assert sr == dataloader.UNIFORM_SAMPLE_RATE, sr
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type = str, nargs = '+', required = True)
    parser.add_argument('--backgrounds', type = str, nargs = '+', required = True)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--clip-duration', type = float, default = 30)
    parser.add_argument('--clips', type = int, default = 2 * 60 * 24)
    parser.add_argument('--bg-change-prob', type = float, default = 0.01666)
    parser.add_argument('--event-prob', type = float, default = 1.0)
    parser.add_argument('--event-freqs', type = str, nargs = '*', default = [])
    parser.add_argument('--max-clusters', type = int, default = 256)
    parser.add_argument('--max-weight', type = float, default = 1024.0)
    parser.add_argument('--filter-thresh', type = float, default = 0.2)
    parser.add_argument('--vote-thresh', type = float, default = 0.0)
    parser.add_argument('--audio-out', type = str)
    parser.add_argument('--fade-duration', type = float, default = 1)
    parser.add_argument('--max-silence-ratio', type = float, default = 0.1)
    parser.add_argument('--radius', type = int, default = 16)
    parser.add_argument('--chunks', type = int, default = 4)
    parser.add_argument('--background-scale', type = float, default = 1.0)
    parser.add_argument('--iterations', type = int, default = 1)
    parser.add_argument('--embedding_size', type = int, default = 16)
    parser.add_argument('--quantized', action = 'store_true')
    parser.add_argument('--quiet', action = 'store_true')
    args = parser.parse_args()

    assert args.iterations >= 1
    if args.seed is not None: random.seed(args.seed)
    globals()['QUIET'] = True

    if args.quantized:
        device = 'cpu'
        print(f'using tflite quantized model on device "{device}"\n')

        encoder = TFLiteModel(model_path = 'model.tflite')
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = 'mps' if torch.backends.mps.is_available() else device

        print(f'using standard model on device "{device}"\n')

        encoder = vae.Encoder(embedding_size = args.embedding_size).to(device)
        encoder.load_state_dict(torch.load('mfcc-8-untested-4/encoder-F16-A0.5-E256-L22.pt', weights_only = True, map_location = device))
        encoder.eval()

    qprint('loading sounds...')
    backgrounds = dict(sum((list(load_sounds(path, min_length = dataloader.SAMPLE_DURATION_SECS, mult_length = dataloader.SAMPLE_DURATION_SECS, max_silence_ratio = args.max_silence_ratio).items()) for path in sorted(args.backgrounds)), start = []))
    events = dict(sum((list(load_sounds(path, max_length = args.clip_duration, max_silence_ratio = args.max_silence_ratio).items()) for path in sorted(args.events)), start = []))
    qprint('loading complete\n')

    qprint(f'backgrounds: {({ k: len(v) for k,v in backgrounds.items() })}')
    qprint(f'events: {({ k: len(v) for k,v in events.items() })}\n')

    # Set up event frequencies
    event_freqs = { x[:x.index(':')]: float(x[x.index(':')+1:]) for x in args.event_freqs }
    if '*' in event_freqs:
        event_freqs = { **event_freqs, **{ x: event_freqs['*'] for x in events.keys() if x not in event_freqs } }
        del event_freqs['*']

    event_freqs = { k: v for k, v in event_freqs.items() if v > 0 }

    if len(event_freqs) == 0: event_freqs = { x: 1 for x in events.keys() }

    # Check for unknown event types
    for x in event_freqs.keys():
        if x not in events:
            raise RuntimeError(f'unknown event type: "{x}"')

    print(f'event freqs: {event_freqs}\n')

    def pick_event() -> str:
        """
        Randomly selects an event based on the defined frequencies.
        """

        t = sum(event_freqs.values())
        r = random.random()
        p = 0
        e = None
        for event, weight in event_freqs.items():
            e = event
            p += weight / t
            if r < p: break
        return e

    # Filter out events that have no frequency defined
    for x in [x for x in events.keys() if x not in event_freqs]:
        del events[x]

    clips = []
    input_events = { x: 0 for x in [None] + list(event_freqs.keys()) }
    output_events = input_events.copy()
    background_class = None
    clip_len = dataloader.UNIFORM_SAMPLE_RATE * args.clip_duration
    
    for i in range(args.iterations):
        f = filter.ClusterFilter(args.max_clusters, args.max_weight, args.embedding_size, args.filter_thresh)

        def vote_retain(x: np.ndarray) -> bool:
            """
            Processes the input audio segment `x` through the filter and returns whether the mean of the votes exceeds the defined threshold.
            """

            votes = []

            chunk_size = dataloader.UNIFORM_SAMPLE_RATE * dataloader.SAMPLE_DURATION_SECS
            for seg in energy_chunks(x, size = chunk_size, count = args.chunks, radius = args.radius):
                with torch.no_grad():
                    spectrogram = mfcc.mfcc_spectrogram_for_learning(seg, dataloader.UNIFORM_SAMPLE_RATE)
                    mean, _ = encoder.forward(torch.tensor(spectrogram[np.newaxis,:], dtype = torch.float).to(device))
                    votes.append(f.insert(mean.cpu().numpy().squeeze()))

            return np.mean(votes) > args.vote_thresh

        # def vote_retain(x: np.ndarray) -> bool:
        #     # Test version that returns randomly True or False
        #     return random.random() < 0.5

        for _ in range(args.clips):
            # Select a random background class if not set or based on the background change probability
            if background_class is None or random.random() < args.bg_change_prob:
                background_class = random.choice(sorted(backgrounds.keys()))
            
            # Select a random background clip
            background = random.choice(backgrounds[background_class])
            clip = np.tile(background, math.ceil(clip_len / len(background)))[:clip_len] # avoid random_contract to prevent transitions in inference chunks
            clip *= args.background_scale
            
            assert clip.shape == (clip_len,), clip.shape

            event_class = None
            # Randomly trigger an event based on the defined probability
            if random.random() < args.event_prob:
                event_class = pick_event()
                event = random.choice(events[event_class])
                event = event * create_fade(len(event), fade_duration = args.fade_duration, sr = dataloader.UNIFORM_SAMPLE_RATE)
                event = random_contract(random_extend(event, clip_len), clip_len)
                clip += event

            if args.audio_out is not None: clips.append(clip)
            input_events[event_class] += 1
            # Apply the filter to the audio clip
            if vote_retain(clip): output_events[event_class] += 1

        # Write the audio clip to file if specified
        if args.audio_out is not None:
            p = args.audio_out if args.iterations == 1 else f'{args.audio_out[:args.audio_out.rfind(".")]}-{i}.{args.audio_out[args.audio_out.rfind(".")+1:]}'
            soundfile.write(args.audio_out, np.concatenate(clips), samplerate = dataloader.UNIFORM_SAMPLE_RATE, format = args.audio_out[args.audio_out.rfind('.')+1:].upper())

    # Print the results
    for event in sorted(input_events.keys(), key = lambda x: -input_events[x]):
        print(f'{str(event):>40}: {input_events[event]:>5} -> {output_events[event]:>5} ({100 * output_events[event] / input_events[event] if input_events[event] != 0 else 0:>5.1f}%)')