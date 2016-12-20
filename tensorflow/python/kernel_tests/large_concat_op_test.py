# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for Concat Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class LargeConcatOpTest(test.TestCase):
  """Tests that belong in concat_op_test.py, but run over large tensors."""

  def testConcatLargeTensors(self):
    # CPU-only test, because it fails on GPUs with <= 4GB memory.
    with ops.device("/cpu:0"):
      a = array_ops.ones([2**31 + 6], dtype=dtypes.int8)
      b = array_ops.zeros([1024], dtype=dtypes.int8)
      onezeros = array_ops.concat_v2([a, b], 0)
    with self.test_session(use_gpu=False):
      # TODO(dga):  Add more depth to this test to validate correctness,
      # not just non-crashingness, once other large tensor fixes have gone in.
      _ = onezeros.eval()


if __name__ == "__main__":
  test.main()
