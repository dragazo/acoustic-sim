import argparse
import dataloader
import random
import math
import soundfile
import numpy as np

from typing import Dict, List, Optional

from sim3util import load_sounds, qprint, create_fade, random_extend, random_contract, energy_chunks, kl_divergence_uniform
from sim3filters import VAEFilter, SpectralFilter, RMSZCFilter, CLAPFilter, make_cluster_filter

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
    parser.add_argument('--background-scale', type = float, default = 0.0)
    parser.add_argument('--iterations', type = int, default = 1)
    parser.add_argument('--embedding_size', type = int, default = 16)
    parser.add_argument('--quantized', action = 'store_true')
    parser.add_argument('--quiet', action = 'store_true')
    parser.add_argument('--filter', type = str, choices = ['vae', 'spectral', 'rmszc', 'clap'], default = 'vae')
    parser.add_argument('--cluster_method', type = str, choices = ['filter', 'filter2'], default = 'filter')
    args = parser.parse_args()
    # Set cluster filter method based on CLI argument
    CLUSTER_METHOD = args.cluster_method

    assert args.iterations >= 1
    if args.seed is not None: random.seed(args.seed)
    globals()['QUIET'] = True

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

    if len(event_freqs) == 0: 
        qprint('No event frequencies specified, using test defaults.')
        test_events = ['Cow', 'Sheep', 'Thunder', 'Aircraft', 'Rooster', 'Frog']
        event_freqs = { x: 1 for x in test_events }

        # Set a random event frequency to 8 times the default
        event_freqs[random.choice(list(event_freqs.keys()))] = 8

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

    # Choose sample rate for filter type
    if args.filter == 'clap':
        sample_rate = 48000
    else:
        sample_rate = dataloader.UNIFORM_SAMPLE_RATE

    clip_len = int(sample_rate * args.clip_duration)

    for i in range(args.iterations):
        # Initialize the filter
        if args.filter == 'vae':
            f = VAEFilter(max_clusters=args.max_clusters, max_weight=args.max_weight, embedding_size=args.embedding_size, filter_thresh=args.filter_thresh, quantized=args.quantized, vote_thresh=args.vote_thresh, radius=args.radius, chunks=args.chunks)
        elif args.filter == 'spectral':
            f = SpectralFilter(max_clusters=args.max_clusters, max_weight=args.max_weight, filter_thresh=args.filter_thresh, clip_len=clip_len)
        elif args.filter == 'rmszc':
            f = RMSZCFilter(max_clusters=args.max_clusters, max_weight=args.max_weight, filter_thresh=args.filter_thresh, clip_len=clip_len)
        elif args.filter == 'clap':
            f = CLAPFilter(max_clusters=args.max_clusters, max_weight=args.max_weight, filter_thresh=args.filter_thresh, clip_len=clip_len, sample_rate=sample_rate)
        else:
            raise ValueError(f'Unknown filter type: {args.filter}')

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
                event = event * create_fade(len(event), fade_duration = args.fade_duration, sr = sample_rate)
                event = random_contract(random_extend(event, clip_len), clip_len)
                clip += event

            if args.audio_out is not None: clips.append(clip)
            input_events[event_class] += 1
            # Apply the filter to the audio clip
            if f.should_retain(clip): output_events[event_class] += 1

        # Write the audio clip to file if specified
        if args.audio_out is not None:
            if args.iterations == 1:
                p = args.audio_out
            else:
                dot = args.audio_out.rfind('.')
                p = f'{args.audio_out[:dot]}-{i}.{args.audio_out[dot+1:]}'
            soundfile.write(p, np.concatenate(clips), samplerate=sample_rate, format=args.audio_out[args.audio_out.rfind('.')+1:].upper())

    # Print the results
    for event in sorted(input_events.keys(), key = lambda x: -input_events[x]):
        print(f'{str(event):>40}: {input_events[event]:>5} -> {output_events[event]:>5} ({100 * output_events[event] / input_events[event] if input_events[event] != 0 else 0:>5.1f}%)')
    
    # Calculate and print the KL divergence
    kl_div = kl_divergence_uniform(output_events)
    print(f'KL divergence from uniform distribution: {kl_div:.4f}')