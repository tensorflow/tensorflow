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

"""Tests to ensure that mlir py_bindings building properly."""

# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
# pylint: disable=consider-using-from-import
# pylint: disable=g-bad-import-order

from pathlib import Path
import sys

print(__file__, file=sys.stderr)
d = Path(__file__).parent


def print_dir(d, level=0):
  print("-" * level, d.name, file=sys.stderr)
  if d.is_dir():
    for f in d.iterdir():
      print_dir(f, level + 1)


print_dir(d)


# Importing the top-level module is fine but doesn't provide submodules.
import tensorflow.compiler.mlir.lite.python.mlir as mlir

assert not hasattr(mlir, "ir")

# Importing a submodule should also be fine.
import tensorflow.compiler.mlir.lite.python.mlir.ir

assert hasattr(mlir, "ir")

# Just some basic API usage to make sure things are roughly in the right place.
context = mlir.ir.Context()
module = mlir.ir.Module.parse("module @foo {}", context)
