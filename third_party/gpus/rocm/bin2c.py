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
"""Simple python utility to represent the contents binary file as an unsigned
char array in C.

This is not meant to be a general purpose tool. On the contrary, it is has only
one, very specific purpose. That is to provide the functionality of the
CUDA bin2c tool, and that too only in the manner it is used during the creation
of MLIR generated kernels.
"""

import os
import argparse


def generate_output(c_file, name, binary_file):
    binary_file_contents = ""
    with open(binary_file, "rb") as f:
        binary_file_contents = f.read()

    size = os.path.getsize(binary_file)
    with open(c_file, "w") as f:
        f.write("static const unsigned char {}[{}] = {{".format(name, size))
        comma = ""
        for i,byte in enumerate(binary_file_contents):
            f.write(comma)
            comma = ","
            if (i % 10) == 0:
                f.write("\n\t")
            f.write("0x{:02x}".format(byte))
        f.write("};\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)
    parser.add_argument("--c_file", default=None)
    parser.add_argument("binary_file")
    args = parser.parse_args()

    generate_output(args.c_file, args.name, args.binary_file)


if __name__ == "__main__":
  main()
