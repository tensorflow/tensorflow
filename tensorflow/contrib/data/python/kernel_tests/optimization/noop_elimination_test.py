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
"""Tests for the MapParallelization optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class NoopEliminationTest(test_base.DatasetTestBase):

  def testNoopElimination(self):
    a = constant_op.constant(1, dtype=dtypes.int64)
    b = constant_op.constant(2, dtype=dtypes.int64)
    some_tensor = math_ops.mul(a, b)

    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.apply(
        optimization.assert_next(
            ["FiniteRepeat", "FiniteSkip", "Prefetch", "Prefetch"]))
    dataset = dataset.repeat(some_tensor).skip(5).prefetch(0).take(-1).skip(
        0).repeat(1).prefetch(0)
    dataset = dataset.apply(optimization.optimize(["noop_elimination"]))

    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      for x in range(5):
        result = sess.run(get_next)
        self.assertAllEqual(result, x)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
