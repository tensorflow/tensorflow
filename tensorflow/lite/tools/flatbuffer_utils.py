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
"""Utility functions for FlatBuffers.

All functions that are commonly used to work with FlatBuffers.

Refer to the tensorflow lite flatbuffer schema here:
tensorflow/lite/schema/schema.fbs

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

from flatbuffers.python import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb

TFLITE_FILE_IDENTIFIER = b'TFL3'


def read_model(input_tflite_file):
  """Reads and parses a tflite model.

  Args:
    input_tflite_file: Full path name to the input tflite file

  Raises:
    RuntimeError: If input_tflite_file is not found.
    IOError: If input_tflite_file cannot be opened.

  Returns:
    A python flatbuffer object corresponding to the input tflite file.
  """
  if not os.path.exists(input_tflite_file):
    raise RuntimeError('Input file not found at %r\n' % input_tflite_file)
  with open(input_tflite_file, 'rb') as file_handle:
    file_data = bytearray(file_handle.read())
  model_obj = schema_fb.Model.GetRootAsModel(file_data, 0)
  return schema_fb.ModelT.InitFromObj(model_obj)


def write_model(model, output_tflite_file):
  """Writes the model, a python flatbuffer object, into the output tflite file.

  Args:
    model: tflite model
    output_tflite_file: Full path name to the output tflite file.

  Raises:
    IOError: If output_tflite_file cannot be opened.
  """
  # Initial size of the buffer, which will grow automatically if needed
  builder = flatbuffers.Builder(1024)
  model_offset = model.Pack(builder)
  builder.Finish(model_offset, file_identifier=TFLITE_FILE_IDENTIFIER)
  model_data = builder.Output()
  with open(output_tflite_file, 'wb') as out_file:
    out_file.write(model_data)


def strip_strings(model):
  """Strips all nonessential strings from the model to reduce model size.

  We remove the following strings:
  (find strings by searching ":string" in the tensorflow lite flatbuffer schema)
  1. Model description
  2. SubGraph name
  3. Tensor names
  We retain OperatorCode custom_code and Metadata name.

  Args:
    model: The model from which to remove nonessential strings.

  """

  model.description = ''
  for subgraph in model.subgraphs:
    subgraph.name = ''
    for tensor in subgraph.tensors:
      tensor.name = ''


def randomize_weights(model, random_seed=0):
  """Randomize weights in a model.

  Args:
    model: The model in which to randomize weights.
    random_seed: The input to the random number generator (default value is 0).

  """

  # The input to the random seed generator. The default value is 0.
  random.seed(random_seed)

  # Parse model buffers which store the model weights
  buffers = model.buffers
  for i in range(1, len(buffers)):  # ignore index 0 as it's always None
    buffer_i_data = buffers[i].data
    buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size

    # Raw data buffers are of type ubyte (or uint8) whose values lie in the
    # range [0, 255]. Those ubytes (or unint8s) are the underlying
    # representation of each datatype. For example, a bias tensor of type
    # int32 appears as a buffer 4 times it's length of type ubyte (or uint8).
    # TODO(b/152324470): This does not work for float as randomized weights may
    # end up as denormalized or NaN/Inf floating point numbers.
    for j in range(buffer_i_size):
      buffer_i_data[j] = random.randint(0, 255)
