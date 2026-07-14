# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Tool to generate C++ headers from LLVM bitcode files.

This tool links multiple LLVM bitcode files using llvm-link and then converts
the resulting bitcode into a C++ header file containing a byte array and an
llvm::StringRef.
"""

import argparse
import itertools
import subprocess


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--llvm_link_bin", required=True, help="Path to the llvm-link binary"
  )
  parser.add_argument(
      "-o", "--output", required=True, help="Output filename for the C++ header"
  )
  parser.add_argument(
      "input_files", nargs="+", help="Variable number of input filenames"
  )
  parser.add_argument(
      "--cpp_namespace",
      default="",
      help="Namespace to be used when generating data",
  )
  parser.add_argument(
      "--cpp_identifier",
      required=True,
      help="Identifier to be used to refer to data",
  )

  args = parser.parse_args()
  llvm_link_bin = args.llvm_link_bin
  output_filename = args.output
  input_filenames = args.input_files
  cpp_namespace = args.cpp_namespace
  cpp_identifier = args.cpp_identifier

  result = subprocess.run(
      [llvm_link_bin, "-f", "-o", "-", "/dev/null"]
      + list(
          itertools.chain.from_iterable(
              ("--override", f) for f in input_filenames
          )
      ),
      capture_output=True,
      check=True,
  )

  llvm_output = result.stdout
  data_string = "".join("\\x{:02x}".format(byte) for byte in llvm_output)

  with open(output_filename, "w") as output_file:
    output_file.write(f"""\
#pragma once

#include "llvm/ADT/StringRef.h"

namespace {cpp_namespace} {{
  inline const char kRaw_{cpp_identifier}[] = "{data_string}";
  constexpr llvm::StringRef {cpp_identifier}{{kRaw_{cpp_identifier}, sizeof(kRaw_{cpp_identifier}) - 1}};
}} // namespace {cpp_namespace}
""")


if __name__ == "__main__":
  main()
