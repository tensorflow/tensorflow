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
r"""Script to generate inputs/outputs exclusion lists for GradientTape.

To use this script:

bazel run tensorflow/python/eager:gen_gradient_input_output_exclusions -- \
  $PWD/tensorflow/python/eager/pywrap_gradient_exclusions.cc
"""

import argparse

from tensorflow.python.eager import gradient_input_output_exclusions


def main(output_file):
  with open(output_file, "w") as fp:
    fp.write(gradient_input_output_exclusions.get_contents())


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("output", metavar="O", type=str, help="Output file.")
  args = arg_parser.parse_args()
  main(args.output)
