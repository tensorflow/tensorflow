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
"""Tests for unary transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.learn.python.learn.dataframe import tensorflow_dataframe as df
from tensorflow.contrib.learn.python.learn.dataframe.transforms.unary_transforms import UNARY_TRANSFORMS
from tensorflow.python.client import session as session_lib
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl

NUMPY_ARRAY_SIZE = 100


class UnaryTestCase(test.TestCase):
  """Test class for unary transforms."""

  @classmethod
  def add_test_case(cls, name, op, np_dtype=float):

    def _test(self):
      if np_dtype == bool:
        arr = np.array([True] * int(NUMPY_ARRAY_SIZE / 2) + [False] * int(
            NUMPY_ARRAY_SIZE / 2))
        np.random.shuffle(arr)
      else:
        arr = np.arange(NUMPY_ARRAY_SIZE, dtype=np_dtype)
      frame = df.TensorFlowDataFrame.from_numpy(
          arr, batch_size=NUMPY_ARRAY_SIZE, shuffle=False)
      self.assertTrue(hasattr(frame["value"], name))
      frame["actual"] = getattr(frame["value"], name)()
      frame_built = frame.build()
      expected_tensor = op(frame_built["value"])
      actual_tensor = frame_built["actual"]

      session = session_lib.Session()
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=session, coord=coord)
      actual, expected = session.run([actual_tensor, expected_tensor])
      coord.request_stop()
      coord.join(threads)
      np.testing.assert_almost_equal(expected, actual)

    setattr(cls, "test{}".format(name), _test)


for ut in UNARY_TRANSFORMS:
  UnaryTestCase.add_test_case(*ut)

if __name__ == "__main__":
  test.main()
