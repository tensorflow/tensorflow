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
"""Functional tests for slice op that consume a lot of GPU memory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SliceTest(test.TestCase):

  def testInt64Slicing(self):
    with self.cached_session(force_gpu=test.is_gpu_available()):
      a_large = array_ops.tile(
          constant_op.constant(np.array([False, True] * 4)), [2**29 + 3])
      slice_t = array_ops.slice(a_large, np.asarray([3]).astype(np.int64), [3])
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([True, False, True], slice_val)

      slice_t = array_ops.slice(
          a_large, constant_op.constant([long(2)**32 + 3], dtype=dtypes.int64),
          [3])
      slice_val = self.evaluate(slice_t)
      self.assertAllEqual([True, False, True], slice_val)
