# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class FilterDatasetTest(test.TestCase):

  def testFilterDataset(self):
    components = (
        np.arange(7, dtype=np.int64),
        np.array([[1, 2, 3]], dtype=np.int64) * np.arange(
            7, dtype=np.int64)[:, np.newaxis],
        np.array(37.0, dtype=np.float64) * np.arange(7)
    )
    count = array_ops.placeholder(dtypes.int64, shape=[])
    modulus = array_ops.placeholder(dtypes.int64)

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
        .repeat(count)
        .filter(lambda x, _y, _z: math_ops.equal(math_ops.mod(x, modulus), 0))
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.test_session() as sess:
      # Test that we can dynamically feed a different modulus value for each
      # iterator.
      def do_test(count_val, modulus_val):
        sess.run(init_op, feed_dict={count: count_val, modulus: modulus_val})
        for _ in range(count_val):
          for i in [x for x in range(7) if x**2 % modulus_val == 0]:
            result = sess.run(get_next)
            for component, result_component in zip(components, result):
              self.assertAllEqual(component[i]**2, result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

      do_test(14, 2)
      do_test(4, 18)

      # Test an empty dataset.
      do_test(0, 1)


if __name__ == "__main__":
  test.main()
