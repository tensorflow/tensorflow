#!/usr/bin/env python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""This tool creates a CPP File that can be embedded in the TF-micro's applications.
Example usage:

python tf_micro_model2cc.py foo.tflite app_model.cc
"""

import json
import os
import shlex
import subprocess
import sys
import binascii

from tensorflow.python.platform import resource_loader
# Schema to use for flatbuffers
_SCHEMA = "third_party/tensorflow/lite/schema/schema.fbs"

# TODO(angerson): fix later when rules are simplified..
_SCHEMA = resource_loader.get_path_to_datafile("../schema/schema.fbs")
_BINARY = resource_loader.get_path_to_datafile("../../../third_party/flatbuffers/flatc")
# Account for different package positioning internal vs. external.
if not os.path.exists(_BINARY):
  _BINARY = resource_loader.get_path_to_datafile(
      "../../../third_party/flatbuffers/flatc")

if not os.path.exists(_SCHEMA):
  raise RuntimeError("Sorry, schema file cannot be found at %r" % _SCHEMA)
if not os.path.exists(_BINARY):
  raise RuntimeError("Sorry, flatc is not available at %r" % _BINARY)

cc_file_prefix = """
// Automatically created by the tf_micro_model2cc.py file

#include "app_model.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif
"""

cc_model_prefix = """
const unsigned char
    g_app_micro_model_data[] DATA_ALIGN_ATTRIBUTE = {
"""
cc_model_suffix = """
};
"""

cc_model_len_string = "const int g_app_micro_model_data_len = {};"

def WriteOpsResolver(output_file, data):
  # Dump the OpsResolver
  ops_resolver_data = """
tflite::MicroMutableOpResolver app_get_model_ops_resolver() {
\ttflite::MicroMutableOpResolver micro_ops_resolver;
"""
  for operator_idx, the_operator in enumerate(data["operator_codes"]):
    ops_resolver_data += "\tmicro_ops_resolver.AddBuiltin(tflite::BuiltinOperator_{}, tflite::ops::micro::Register_{}());\n".format(the_operator["builtin_code"], the_operator["builtin_code"])
  ops_resolver_data += "return micro_ops_resolver;\n}\n"
  output_file.write(ops_resolver_data)

def WriteModelData(output_file, tflite_input):
  # Dump the Model data
  chunk_size = 1200
  total_size = 0
  with open (tflite_input, 'rb') as f:
    while True:
      data = f.read(chunk_size)
      if not data:
        break
      total_size += len(data)
      hex_data = binascii.hexlify(data)
      hex_string = hex_data.decode("ascii")
      for i in range(0, len(hex_string), 2):
        if (i % 24 == 0):
          output_file.write("\n\t")
        output_file.write("0x{}{}, ".format(hex_string[i], hex_string[i + 1]))

  return total_size

def CreateCCFile(tflite_input, cc_output):
  """Given a tflite model in `tflite_input` file, produce html description."""

  # Convert the model into a JSON flatbuffer using flatc (build if doesn't
  # exist.
  if not os.path.exists(tflite_input):
    raise RuntimeError("Invalid filename %r" % tflite_input)
  if tflite_input.endswith(".tflite") or tflite_input.endswith(".bin"):

    # Run convert
    cmd = (
        _BINARY + " -t "
        "--strict-json --defaults-json -o /tmp {schema} -- {input}".format(
            input=tflite_input, schema=_SCHEMA))
    print(cmd)
    subprocess.check_call(shlex.split(cmd))
    real_output = ("/tmp/" + os.path.splitext(
        os.path.split(tflite_input)[-1])[0] + ".json")

    data = json.load(open(real_output))
  else:
    raise RuntimeError("Input file was not .tflite or .json")

  output_file = open(cc_output, "w")
  output_file.write(cc_file_prefix)
  WriteOpsResolver(output_file, data)
  output_file.write(cc_model_prefix)
  model_size = WriteModelData(output_file, tflite_input)
  output_file.write(cc_model_suffix)
  output_file.write(cc_model_len_string.format(model_size))

      
      
def main(argv):
  try:
    tflite_input = argv[1]
    cc_output = argv[2]
  except IndexError:
    print("Usage: %s <input tflite> <output cc>" % (argv[0]))
  else:
    CreateCCFile(tflite_input, cc_output)


if __name__ == "__main__":
  main(sys.argv)
