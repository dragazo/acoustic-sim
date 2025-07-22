import subprocess
import argparse
import itertools
import threading
import os

EVENTS = [
    # '/home/devin/Downloads/dog-events/',
    # '/home/devin/Downloads/misc-events/',
    './misc-events/',
]
BACKGROUNDS = [
    #'/home/devin/Downloads/edansa-events/',
    './edansa-events/',
]

# -------------------------------------------------------

EVENT_FREQS = [
    # [0, 'Frog:8', 'Rooster:1', 'Aircraft:1'],
    [1, 'Frog:1', 'Rooster:8', 'Aircraft:1'],
]
MAX_CLUSTERS = [
    # 8,
    # 12,
    # 16,
    # 24,
    # 32,
    # 40,
    # 48,
    # 56,
    # 64,
    # 72,
    # 80,
    # 88,
    # 96,
    # 104,
    # 112,
    # 120,
    # 128,
    # 132,
    # 140,
    # 148,
    # 156,
    # 164,
    # 172,
    # 180,
    # 188,
    # 196,
    204,
    212,
    220,
    228,
    236,
    244,
    252,
    260,
    268,
    276,
    284,
    292,
    300,
    # 256,
    # 512,
    # 1024,
]
MAX_WEIGHT = [
    # 8,
    # 16,
    # 32,
    # 40,
    # 48,
    # 56,
    64,
    # 72,
    # 80,
    # 88,
    # 96,
    # 104,
    # 112,
    # 120,
    128,
    # 192,
    256,
    512,
    # 768,
    1024,
    # 1536,
    2048,
]
FILTER_THRESH = [
    # 0.005,
    # 0.010,
    # 0.015,
    # 0.018,
    0.019,
    # 0.020,
    0.021,
    # 0.022,
    0.023,
    # 0.024,
    0.025,
    # 0.026,
    # 0.027,
    # 0.028,
    # 0.029,
    # 0.030,
    # 0.031,
    # 0.032,
    # 0.033,
    # 0.034,
    # 0.035,
    # 0.040,
    # 0.045,
    # 0.050,
    # 0.055,
    # 0.060,
    # 0.065,
    # 0.070,
    # 0.075,
    # 0.080,
    # 0.085,
    # 0.090,
    # 0.095,
    # 0.100,
    # 0.105,
    # 0.110,
    # 0.115,
    # 0.120,
    # 0.125,
    # 0.130,
    # 0.135,
    # 0.140,
    # 0.145,
    # 0.150,
    # 0.2,
    # 0.25,
    # 0.3,
    # 0.35,
    # 0.4,
    # 0.45,
    # 0.5,
    # 0.55,
    # 0.6,
    # 0.65,
    # 0.7,
]
VOTE_THRESH = [
    # 0.0,
    # 0.1,
    # 0.2,
    0.3,
    # 0.4,
    # 0.5,
    0.6,
]
RADIUS = [
    8,
    # 16,
    # 24,
    # 32,
    # 40,
    # 64,
    # 128,
    # 256,
]
CHUNKS = [
    # 0,
    4,
    # 8,
    # 12,
    # 16,
    # 20,
    # 24,
    # 28,
]
BACKGROUND_SCALE = [
    0.0,
    # 1.0,
]

# -------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type = int, required = True)
    parser.add_argument('--iterations', type = int, default = 1)
    parser.add_argument('--hours', type = float, required = True)
    parser.add_argument('--output', type = str, required = True)
    parser.add_argument('--quantized', action = 'store_true')
    args = parser.parse_args()

    assert args.jobs >= 1
    assert args.hours >= 0.1

    if not os.path.exists(args.output): os.mkdir(args.output)

    queue = list(itertools.product(EVENT_FREQS, MAX_CLUSTERS, MAX_WEIGHT, FILTER_THRESH, VOTE_THRESH, RADIUS, CHUNKS, BACKGROUND_SCALE))
    queue_init_size = len(queue)
    queue_lock = threading.Lock()
    def worker():
        while True:
            with queue_lock:
                if len(queue) == 0: return
                i = queue_init_size - len(queue)
                info = queue.pop()
                print(f'starting task {i + 1}/{queue_init_size}')
            file = f'{args.output}/b{info[7]}-p{info[6]}-r{info[5]}-v{info[4]}-c{info[1]}-w{info[2]}-t{info[3]}-e{info[0][0]}.txt'
            if os.path.exists(file): continue
            with open(file, 'w') as f:
                subprocess.check_call([
                    'time', 'python', 'sim2.py', '--quiet',
                    '--events', *EVENTS,
                    '--backgrounds', *BACKGROUNDS,
                    '--clips', str(round(2 * 60 * args.hours)),
                    '--iterations', str(args.iterations),
                    '--event-prob', '1.0',
                    '--event-freqs', *info[0][1:],
                    '--max-clusters', str(info[1]),
                    '--max-weight', str(info[2]),
                    '--filter-thresh', str(info[3]),
                    '--vote-thresh', str(info[4]),
                    '--radius', str(info[5]),
                    '--chunks', str(info[6]),
                    '--background-scale', str(info[7]),
                    *(['--quantized'] if args.quantized else []),
                ], stdout = f, stderr = f)
    workers = [threading.Thread(target = worker) for _ in range(args.jobs)]
    for w in workers: w.start()
    for w in workers: w.join()
