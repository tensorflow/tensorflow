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
"""Coder operations tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.coder.python.ops import coder_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class CoderOpsTest(test.TestCase):
  """Coder ops test.

  Coder ops have C++ tests. Python test just ensures that Python binding is not
  broken.
  """

  def testReadmeExample(self):
    data = random_ops.random_uniform((128, 128), 0, 10, dtype=dtypes.int32)
    histogram = math_ops.bincount(data, minlength=10, maxlength=10)
    cdf = math_ops.cumsum(histogram, exclusive=False)
    cdf = array_ops.pad(cdf, [[1, 0]])
    cdf = array_ops.reshape(cdf, [1, 1, -1])

    data = math_ops.cast(data, dtypes.int16)
    encoded = coder_ops.range_encode(data, cdf, precision=14)
    decoded = coder_ops.range_decode(
        encoded, array_ops.shape(data), cdf, precision=14)

    with self.cached_session() as sess:
      self.assertAllEqual(*sess.run((data, decoded)))


if __name__ == '__main__':
  test.main()
