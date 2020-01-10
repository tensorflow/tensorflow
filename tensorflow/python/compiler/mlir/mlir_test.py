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
# =============================================================================
"""Tests for python.compiler.mlir."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.compiler.mlir import mlir
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class MLIRImportTest(test.TestCase):

  def test_import_graph_def(self):
    """Tests the basic flow of `tf.mlir.experimental.convert_graph_def`."""
    mlir_module = mlir.convert_graph_def('')
    # An empty graph should contain at least an empty main function.
    self.assertIn('func @main', mlir_module)

  def test_invalid_pbtxt(self):
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 'Could not parse input proto'):
      mlir.convert_graph_def('some invalid proto')


if __name__ == '__main__':
  test.main()
