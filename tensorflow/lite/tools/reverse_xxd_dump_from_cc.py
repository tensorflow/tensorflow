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
r"""This tool reverses xxd dump from *.cc source file to its original binary file

This script is used to convert models from C++ source file (dumped with xxd) to
the binary model weight file and analyze it with model visualizer like Netron
(https://github.com/lutzroeder/netron) or load the model in TensorFlow Python API
to evaluate the results in Python.

The command to dump binary file to C++ source file looks like

xxd -i model_data.tflite > model_data.cc

Example usage:

python reverse_xxd_dump_from_cc.py model_data.cc --output=model_data.tflite
"""
import argparse
import os
import re


def generate_default_output(filename, postfix=None, extension=None):
  """Generate output filename given the filename and extension

  Args:
    filename(str): Input filename
    postfix(str): Postfix to add to the output filename
    extension(str): Output file extension, if not given, it will be
      the same as input file.

  Return:
    string for the output filename given input args
  """
  name, ext = os.path.splitext(filename)

  if extension is not None:
    if not extension.startswith("."):
      extension = "." + extension

    ext = extension

  if postfix is None:
    postfix = ""

  output = "{}{}{}".format(name, postfix, ext)

  return output


def reverse_dump(filename, output=None, extension=".tflite"):
  """Reverse dump the tensorflow model weight from C++ array source array

  Args:
    filename(str): Input filename (the input *.cc file)
    output(str): Output filename, default to be same as input file but
      with different extension, default extension is *.tflite
  """
  if output is None:
    output = generate_default_output(filename, extension=extension)

  # Pattern to match with hexadecimal value in the array
  pattern = re.compile(r"\W*(0x[0-9a-fA-F,x ]+).*")

  array = bytearray()
  with open(filename) as f:
    for line in f:
      values_match = pattern.match(line)

      if values_match is None:
        continue

      # Match in the parentheses (hex array only)
      list_text = values_match.group(1)
      # Extract hex values (text)
      values_text = filter(None, list_text.split(","))
      # Convert to hex
      values = [int(x, base=16) for x in values_text]

      array.extend(values)

  with open(output, 'wb') as f:
    f.write(array)

  print("Byte data written to `{}`".format(output))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "source",
    type=str,
    help="C/C++ source file dumped from `xxd -i [HEX_FILE]`")
  parser.add_argument("-o", "--output", type=str, help="Output filename")

  args = parser.parse_args()

  reverse_dump(args.source, args.output)
