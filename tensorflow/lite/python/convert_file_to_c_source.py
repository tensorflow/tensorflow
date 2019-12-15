# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Python command line interface for converting TF Lite files into C source."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.lite.python import util
from tensorflow.python.platform import app


def run_main(_):
  """Main in convert_file_to_c_source.py."""

  parser = argparse.ArgumentParser(
      description=("Command line tool to run TensorFlow Lite Converter."))

  parser.add_argument(
      "--input_tflite_file",
      type=str,
      help="Full filepath of the input TensorFlow Lite file.",
      required=True)

  parser.add_argument(
      "--output_source_file",
      type=str,
      help="Full filepath of the output C source file.",
      required=True)

  parser.add_argument(
      "--output_header_file",
      type=str,
      help="Full filepath of the output C header file.",
      required=True)

  parser.add_argument(
      "--array_variable_name",
      type=str,
      help="Name to use for the C data array variable.",
      required=True)

  parser.add_argument(
      "--line_width", type=int, help="Width to use for formatting.", default=80)

  parser.add_argument(
      "--include_guard",
      type=str,
      help="Name to use for the C header include guard.",
      default=None)

  parser.add_argument(
      "--include_path",
      type=str,
      help="Optional path to include in generated source file.",
      default=None)

  parser.add_argument(
      "--use_tensorflow_license",
      dest="use_tensorflow_license",
      help="Whether to prefix the generated files with the TF Apache2 license.",
      action="store_true")
  parser.set_defaults(use_tensorflow_license=False)

  flags, _ = parser.parse_known_args(args=sys.argv[1:])

  with open(flags.input_tflite_file, "rb") as input_handle:
    input_data = input_handle.read()

  source, header = util.convert_bytes_to_c_source(
      data=input_data,
      array_name=flags.array_variable_name,
      max_line_width=flags.line_width,
      include_guard=flags.include_guard,
      include_path=flags.include_path,
      use_tensorflow_license=flags.use_tensorflow_license)

  with open(flags.output_source_file, "w") as source_handle:
    source_handle.write(source)

  with open(flags.output_header_file, "w") as header_handle:
    header_handle.write(header)


def main():
  app.run(main=run_main, argv=sys.argv[:1])


if __name__ == "__main__":
  main()
