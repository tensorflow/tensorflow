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
"""Randomize all weights in a tflite file.

Example usage:
python randomize_weights.py foo.tflite foo_randomized.tflite
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.platform import app


def main(_):
  parser = argparse.ArgumentParser(
      description='Randomize weights in a tflite file.')
  parser.add_argument(
      '--input_tflite_file',
      type=str,
      required=True,
      help='Full path name to the input tflite file.')
  parser.add_argument(
      '--output_tflite_file',
      type=str,
      required=True,
      help='Full path name to the output randomized tflite file.')
  parser.add_argument(
      '--random_seed',
      type=str,
      required=False,
      default=0,
      help='Input to the random number generator. The default value is 0.')
  args = parser.parse_args()

  # Read the model
  input_model = flatbuffer_utils.read_model(args.input_tflite_file)
  # Invoke the randomize weights function
  output_model = flatbuffer_utils.randomize_weights(input_model,
                                                    args.random_seed)
  # Write the model
  flatbuffer_utils.write_model(output_model, args.output_tflite_file)


if __name__ == '__main__':
  app.run(main=main, argv=sys.argv[:1])
