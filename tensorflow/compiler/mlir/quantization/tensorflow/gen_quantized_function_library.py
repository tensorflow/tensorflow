# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Generates the quantized function library contained header file."""

from typing import Sequence

from absl import app
from absl import flags

_OUTPUT_FILE = flags.DEFINE_string('output_file', None, 'output file location')
_SRC = flags.DEFINE_string('src', None, 'source file location')

flags.mark_flags_as_required(['output_file', 'src'])


def main(_: Sequence[str]) -> None:
  with open(_SRC.value, 'r') as f:
    lines = f.readlines()

  with open(_OUTPUT_FILE.value, 'w') as f:
    f.write("""/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_QUANTIZED_FUNCTION_LIBRARY_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_QUANTIZED_FUNCTION_LIBRARY_H_

namespace mlir {
namespace quant {

constexpr char kQuantizedFunctionLibraryInMLIR[] =""")

    for line in lines:
      f.write('\n    "')
      f.write(line.rstrip().replace('"', r'\"'))
      f.write('\\n"')

    f.write(""";

}  // namespace quant
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_QUANTIZED_FUNCTION_LIBRARY_H_
""")


if __name__ == '__main__':
  app.run(main)
