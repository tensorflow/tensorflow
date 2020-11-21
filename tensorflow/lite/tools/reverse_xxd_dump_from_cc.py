# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Reverses xxd dump from to binary file

This script is used to convert models from C++ source file (dumped with xxd) to
the binary model weight file and analyze it with model visualizer like Netron
(https://github.com/lutzroeder/netron) or load the model in TensorFlow Python
API
to evaluate the results in Python.

The command to dump binary file to C++ source file looks like

xxd -i model_data.tflite > model_data.cc

Example usage:

python reverse_xxd_dump_from_cc.py \
  --input_cc_file=model_data.cc \
  --output_tflite_file=model_data.tflite
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.platform import app


def main(_):
  """Application run loop."""
  parser = argparse.ArgumentParser(
      description='Reverses xxd dump from to binary file')
  parser.add_argument(
      '--input_cc_file',
      type=str,
      required=True,
      help='Full path name to the input cc file.')
  parser.add_argument(
      '--output_tflite_file',
      type=str,
      required=True,
      help='Full path name to the stripped output tflite file.')

  args = parser.parse_args()

  # Read the model from xxd output C++ source file
  model = flatbuffer_utils.xxd_output_to_object(args.input_cc_file)
  # Write the model
  flatbuffer_utils.write_model(model, args.output_tflite_file)


if __name__ == '__main__':
  app.run(main=main, argv=sys.argv[:1])
