# Lint as: python3
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
r"""Modify a quantized model's interface from float to integer.

Example usage:
python modify_model_interface_main.py \
  --input_file=float_model.tflite \
  --output_file=int_model.tflite \
  --input_type=INT8 \
  --output_type=INT8
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.lite.tools.optimize.python import modify_model_interface_constants as mmi_constants
from tensorflow.lite.tools.optimize.python import modify_model_interface_lib as mmi_lib
from tensorflow.python.platform import app


def main(_):
  """Application run loop."""
  parser = argparse.ArgumentParser(
      description="Modify a quantized model's interface from float to integer.")
  parser.add_argument(
      '--input_file',
      type=str,
      required=True,
      help='Full path name to the input tflite file.')
  parser.add_argument(
      '--output_file',
      type=str,
      required=True,
      help='Full path name to the output tflite file.')
  parser.add_argument(
      '--input_type',
      type=str.upper,
      choices=mmi_constants.STR_TYPES,
      default=mmi_constants.DEFAULT_STR_TYPE,
      help='Modified input integer interface type.')
  parser.add_argument(
      '--output_type',
      type=str.upper,
      choices=mmi_constants.STR_TYPES,
      default=mmi_constants.DEFAULT_STR_TYPE,
      help='Modified output integer interface type.')
  args = parser.parse_args()

  input_type = mmi_constants.STR_TO_TFLITE_TYPES[args.input_type]
  output_type = mmi_constants.STR_TO_TFLITE_TYPES[args.output_type]

  mmi_lib.modify_model_interface(args.input_file, args.output_file, input_type,
                                 output_type)

  print('Successfully modified the model input type from FLOAT to '
        '{input_type} and output type from FLOAT to {output_type}.'.format(
            input_type=args.input_type, output_type=args.output_type))


if __name__ == '__main__':
  app.run(main=main, argv=sys.argv[:1])
