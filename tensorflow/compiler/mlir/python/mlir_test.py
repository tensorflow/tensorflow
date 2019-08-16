# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the Python extension-based XLA client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.mlir.python.mlir_extension import import_graphdef
from tensorflow.python.platform import test


class MLIRConvertTest(test.TestCase):

  def testEmptyPbtxtToMlir(self):
    mlir_module = import_graphdef("")
    self.assertIn("func @main", mlir_module)

  def testInvalidPbtxtToMlir(self):
    with self.assertRaisesRegexp(RuntimeError,
                                 "Error parsing proto, see logs for error"):
      import_graphdef("some invalid proto")


if __name__ == "__main__":
  test.main()
