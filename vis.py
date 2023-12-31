import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np

import argparse
import json

from util import load_sound

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('labels')
parser.add_argument('-e', '--event', type = int)
parser.add_argument('-s', '--spectrum', action = 'store_true')
args = parser.parse_args()

def clamp(x, minimum, maximum):
    return max(min(x, maximum), minimum)

with open(args.labels, 'r') as f:
    labels = json.load(f)

source = load_sound(args.source)
sample_rate, samples = wav.read(args.source)
channels = [samples] if len(samples.shape) == 1 else [samples[:,i] for i in range(samples.shape[1])]
for channel in channels:
    assert len(channel.shape) == 1
    assert channel.shape == channels[0].shape

if args.event is not None:
    event = labels[args.event]
    start = clamp(round(event.get('start', 0) * sample_rate), 0, len(channel))
    stop = clamp(start + round(event.get('duration', 0) * sample_rate), start, len(channel))
    channels = [channel[start:stop] for channel in channels]
    title = f'{args.source} ({sample_rate}Hz) - event {args.event} ({event["kind"]})'
else:
    title = f'{args.source} ({sample_rate}Hz)'

if args.spectrum:
    combined = np.average(channels, axis = 0)
    f, t, Sxx = sig.spectrogram(combined, sample_rate)
    plt.pcolormesh(t, f, np.log10(Sxx))
    plt.ylabel('frequency (Hz)')
else:
    for i, channel in enumerate(channels):
        plt.plot(np.arange(len(channel)) / sample_rate, channel, label = f'channel {i + 1}')
    plt.ylabel('amplitude')
    plt.legend(loc = 'upper left')

plt.title(title)
plt.xlabel('time (s)')
plt.show()
