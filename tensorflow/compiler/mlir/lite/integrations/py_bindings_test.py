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


def smoketest():
  import tensorflow.compiler.mlir.lite.integrations.python.mlir


if __name__ == "__main__":
  smoketest()
