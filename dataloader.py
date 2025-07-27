import librosa
import random
import sys
import os

import scipy.signal as sig
import numpy as np

from typing import Optional

DATASET_EVENT_DURATION_SECS = 2
SAMPLE_DURATION_SECS = 1
UNIFORM_SAMPLE_RATE = 8000 # 20000

def load_dataset_file(path: str, *, sample_duration: int = SAMPLE_DURATION_SECS, sample_rate: int = UNIFORM_SAMPLE_RATE):
    p = path.rfind('.')
    if p >= 0: path = path[:p]

    try:
        orig_samples, orig_sr = librosa.load(f'{path}.wav', sr = None)
        if orig_sr != sample_rate:
            new_samples = librosa.resample(orig_samples, orig_sr = orig_sr, target_sr = sample_rate)
        else:
            new_samples = orig_samples
    except Exception as e:
        print(f'failed to read {path}.wav', file = sys.stderr)
        raise e

    raw_events = []
    with open(f'{path}.meta', 'r') as f:
        for line in f:
            vals = [x.strip() for x in line.strip().split(',')]
            if len(vals) == 2:
                vals = ['TRAIN', *vals]
            assert len(vals) == 3
            if vals[2].lower() == 'ignore': continue
            vals[1] = round(int(vals[1]) * (sample_rate / orig_sr))
            raw_events.append(vals)
    raw_events.sort(key = lambda x: x[1])

    res = []
    t = 0
    dt = round(sample_duration * sample_rate)
    edt = round(DATASET_EVENT_DURATION_SECS * sample_rate)
    while True:
        if t + dt > new_samples.shape[0]: break
        label = 'BackgroundSounds'
        for raw_event in raw_events:
            if t + dt > raw_event[1] and t < raw_event[1] + edt:
                label = raw_event[2]
                break
        res.append((label, new_samples[t:t+dt]))
        t += dt
    return res

def get_dataset(max_files_per_class: Optional[int], max_events_per_class: Optional[int], classes: Optional[list[str]], classes_strict: bool = False, *, sample_duration: int = SAMPLE_DURATION_SECS, sample_rate: int = UNIFORM_SAMPLE_RATE):
    """
    Get a dataset of audio samples from the 'dataset-partial' directory.

    Args:
        max_files_per_class (Optional[int]): Maximum number of files to load per class.
        max_events_per_class (Optional[int]): Maximum number of events to load per class.
        classes (Optional[list[str]]): List of class labels to include. If None, all classes are included.
        classes_strict (bool): If False, classes are filtered during loading, which is faster but may miss some samples.
        sample_duration (int): Duration of each sample in seconds.
        sample_rate (int): Sample rate for audio processing.

    Returns:
        dict: A dictionary where keys are class labels and values are lists of audio samples.
    """

    if not os.path.exists('dataset-partial'):
        raise FileNotFoundError("The 'dataset-partial' directory does not exist. Please ensure the dataset is available.")

    dataset = {}
    root = 'dataset-partial'

    for cls in os.listdir(root):
        if not classes_strict and classes is not None and cls not in classes:
            continue
        files = []
        for dirpath, dirnames, filenames in os.walk(f'{root}/{cls}'):
            for filename in filenames:
                if filename.endswith('.wav'):
                    files.append(f'{dirpath}/{filename}')
        
        random.shuffle(files)

        for i, file in enumerate(files):
            if max_files_per_class is not None and i >= max_files_per_class: 
                break

            sub_dataset = {}

            for label, data in load_dataset_file(file, sample_duration = sample_duration, sample_rate = sample_rate):
                if classes is not None and label not in classes:
                    continue

                if label not in sub_dataset:
                    sub_dataset[label] = []
                sub_dataset[label].append(data)

            for k, v in sub_dataset.items():
                random.shuffle(v)
                if k not in dataset:
                    dataset[k] = []
                dataset[k].extend(v)

    for k in dataset.keys():
        random.shuffle(dataset[k])

        if max_events_per_class is not None and len(dataset[k]) > max_events_per_class:
            dataset[k] = dataset[k][:max_events_per_class]
    
    return dataset

# def get_dataset(max_files_per_class, max_events_per_class):
#     dataset = {}
#     for dirpath, dirnames, filenames in os.walk('dataset-partial'):
#         random.shuffle(filenames)
#         for i, filename in enumerate(filenames):
#             if i >= max_files_per_class: break
#             if not filename.endswith('.wav'): continue
#             sub_dataset = {}
#             for label, data in load_dataset_file(f'{dirpath}/{filename}'):
#                 if label not in sub_dataset:
#                     sub_dataset[label] = []
#                 sub_dataset[label].append(data)
#             for k, v in sub_dataset.items():
#                 random.shuffle(v)
#                 if k not in dataset:
#                     dataset[k] = []
#                 dataset[k].extend(v)
#     for k in dataset.keys():
#         random.shuffle(dataset[k])
#         if len(dataset[k]) > max_events_per_class:
#             dataset[k] = dataset[k][:max_events_per_class]
#     return dataset

def get_spectrogram_normalized(sample, *, any_channel = False, sample_rate: int = UNIFORM_SAMPLE_RATE):
    """
    Compute a normalized spectrogram for a given audio sample.

    Args:
        sample (np.ndarray): Audio sample.
        any_channel (bool): If True, randomly select a channel if multi-channel.
        sample_rate (int): Sample rate of the audio.

    Returns:
        tuple: (spectrum, normalization parameters)
    """
    if len(sample.shape) == 2 and any_channel:
        assert sample.shape[0] > 0
        sample = random.choice(sample)
    assert len(sample.shape) == 1

    sample_offset = np.mean(sample)
    sample -= sample_offset

    sample_scale = np.sqrt(np.sum(sample**2))
    if sample_scale <= 0:
        sample_scale = 1
    sample /= sample_scale

    f, t, Sxx = sig.spectrogram(sample, sample_rate)
    spectrum = np.log10(np.maximum(Sxx, 1e-20))
    # spectrum = Sxx

    spectrum_offset = 0 #np.min(spectrum)
    spectrum -= spectrum_offset

    spectrum_scale = 1 #np.mean(spectrum)
    if spectrum_scale <= 0:
        spectrum_scale = 1
    spectrum /= spectrum_scale

    return spectrum, (spectrum_scale, spectrum_offset, sample_scale, sample_offset)
