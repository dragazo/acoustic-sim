import argparse
import json
import uuid
import sys
import os

from pydub import AudioSegment
from typing import List, Any
from util import load_sound

parser = argparse.ArgumentParser()
parser.add_argument('sim')
parser.add_argument('--out', default = 'out.wav')
parser.add_argument('--sr', type = int)
args = parser.parse_args()

if args.sim == '-':
    sim = json.load(sys.stdin)
else:
    with open(args.sim, 'r') as f:
        sim = json.load(f)

def apply_effects(sound: AudioSegment, effects: List[Any]) -> AudioSegment:
    for effect in effects:
        if not isinstance(effect, list):
            effect = [effect]

        if effect[0] == 'gain':
            sound = sound + effect[1]
        elif effect[0] == 'slice':
            sound = sound[effect[1] * 1000 : effect[2] * 1000]
        elif effect[0] == 'mp3':
            temp = f'__temp_cvt__{uuid.uuid4().hex}.mp3'
            sound.export(temp, format = 'mp3')
            sound = load_sound(temp)
            os.remove(temp)
        else:
            raise RuntimeError(f'unknown effect type: {effect[0]}')
    return sound

duration = 0
sounds = []
for event in sim['events']:
    sound = apply_effects(load_sound(event['source']), event.get('effects', []))
    sounds.append(sound)
    duration = max(duration, event.get('start', 0) + sound.duration_seconds)

res = AudioSegment.silent(duration * 1000)
for i, event in enumerate(sim['events']):
    sound = sounds[i]
    res = res.overlay(sound, event.get('start', 0) * 1000)
res = apply_effects(res, sim.get('effects', []))

export_params = []
if args.sr is not None:
    export_params.extend(['-ar', str(args.sr)])
res.export(args.out, format = 'wav', parameters = export_params)
