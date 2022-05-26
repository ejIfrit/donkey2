#!/usr/bin/env python3
"""
Scripts to play back tubs and see what the model is doing.
Basic usage should feel familiar: playback.py --tubs data/ --model models/mypilot.h5

Usage:
    playback.py [--tubs=tubs] (--model=<model>)
    [--type=(linear|inferred|tensorrt_linear|tflite_linear)]
    [--comment=<comment>]

Options:
    -h --help              Show this screen.
"""

from docopt import docopt
import donkeycar as dk
from donkeycar.management.playback_ej import playBackClass


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    comment = args['--comment']
    pb = playBackClass()
    pb.run(cfg, tubs, model, model_type)


if __name__ == "__main__":
    main()
