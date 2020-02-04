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
"""This tool strips all nonessential strings from a tflite file.

Refer to the schema here: //third_party/tensorflow/lite/schema/schema.fbs
We remove the following strings: (search for ":string" in this schema)
1. Tensor names
2. SubGraph name
3. Model description
We retain OperatorCode custom_code and Metadata name.

Example usage:

python strip_strings.py foo.tflite foo_stripped.tflite
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from flatbuffers.python import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.python.platform import app


def StripTfliteFile(input_tflite_file, output_tflite_file):
  """Strips all nonessential strings from the model to reduce model size.

  Args:
    input_tflite_file: Full path name to the input tflite file
    output_tflite_file: Full path name to the stripped output tflite file.

  Raises:
    RuntimeError: If input_tflite_file is not found.
    IOError: If input_tflite_file or output_tflite_file cannot be opened.

  """

  if not os.path.exists(input_tflite_file):
    raise RuntimeError('Input file not found at %r\n' % input_tflite_file)
  with open(input_tflite_file, 'rb') as file_handle:
    file_data = bytearray(file_handle.read())
  model_obj = schema_fb.Model.GetRootAsModel(file_data, 0)
  model = schema_fb.ModelT.InitFromObj(model_obj)
  model.description = ''
  for subgraph in model.subgraphs:
    subgraph.name = ''
    for tensor in subgraph.tensors:
      tensor.name = ''
  builder = flatbuffers.Builder(1024)  # Initial size of the buffer, which
  # will grow automatically if needed
  model_offset = model.Pack(builder)
  builder.Finish(model_offset)
  model_data = builder.Output()
  with open(output_tflite_file, 'wb') as out_file:
    out_file.write(model_data)


def main(_):
  """Application run loop."""
  parser = argparse.ArgumentParser(
      description='Strips all nonessential strings from a tflite file.')
  parser.add_argument(
      'input_tflite_file',
      type=str,
      help='Full path name to the input tflite file.')
  parser.add_argument(
      'output_tflite_file',
      type=str,
      help='Full path name to the stripped output tflite file.')
  args = parser.parse_args()

  # Invoke the strip tflite file function
  StripTfliteFile(args.input_tflite_file, args.output_tflite_file)


if __name__ == '__main__':
  app.run(main=main, argv=sys.argv[:1])
