#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" Transform dumped step stats to chrome formatted timeline json file
"""

import argparse
import json
import pdb

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.core.framework.step_stats_pb2 import StepStats

def main():
    parser = argparse.ArgumentParser(description='Transform dumped step stats to chrome formatted timeline json file')
    parser.add_argument('step_stats', help='dumped step stats file')
    parser.add_argument('output', help='json file to save to')
    args = parser.parse_args()
    with open(args.step_stats) as step_stats_file:
        step_stats = StepStats()
        step_stats.ParseFromString(step_stats_file.read())
    trace = timeline.Timeline(step_stats=step_stats)
    with open(args.output, 'w') as output_file:
        output_file.write(trace.generate_chrome_trace_format())

if __name__ == '__main__':
    main()
