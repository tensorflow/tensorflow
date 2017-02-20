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
"""Tests for binary transforms."""

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
from tensorflow.contrib.learn.python.learn.dataframe.transforms.binary_transforms import BINARY_TRANSFORMS
from tensorflow.python.client import session as session_lib
from tensorflow.python.platform import test as test_lib
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl

NUMPY_ARRAY_SIZE = 100
SCALAR = 50.0
TEST_NAME_PREFIX = "testBinaryOp_"


class BinaryTransformTestCase(test_lib.TestCase):
  """Test class for binary transforms."""

  @classmethod
  def add_test_case(cls, fn_name, op):

    def _test(self):
      rng = np.arange(
          -NUMPY_ARRAY_SIZE // 2, NUMPY_ARRAY_SIZE // 2, dtype="float32")

      frame = df.TensorFlowDataFrame.from_numpy(
          rng, batch_size=len(rng), shuffle=False)

      frame["sqr"] = frame["value"].square()

      self.assertTrue(hasattr(frame["value"], fn_name))

      frame["series_result"] = getattr(frame["value"], fn_name)(frame["sqr"])
      frame["scalar_result"] = getattr(frame["value"], fn_name)(SCALAR)

      frame_built = frame.build()

      expected_series_tensor = op(frame_built["value"], frame_built["sqr"])
      actual_series_tensor = frame_built["series_result"]

      expected_scalar_tensor = op(frame_built["value"], SCALAR)
      actual_scalar_tensor = frame_built["scalar_result"]

      session = session_lib.Session()
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=session, coord=coord)
      actual_series, expected_series, actual_scalar, expected_scalar = (
          session.run([
              actual_series_tensor, expected_series_tensor,
              actual_scalar_tensor, expected_scalar_tensor
          ]))
      coord.request_stop()
      coord.join(threads)
      np.testing.assert_almost_equal(expected_series, actual_series)
      np.testing.assert_almost_equal(expected_scalar, actual_scalar)

    setattr(cls, "{}{}".format(TEST_NAME_PREFIX, op.__name__), _test)


for bt in BINARY_TRANSFORMS:
  BinaryTransformTestCase.add_test_case(*bt)

# Check that the number of test methods matches the number of binary transforms.
test_methods = [
    test for test in dir(BinaryTransformTestCase)
    if test.startswith(TEST_NAME_PREFIX)
]
assert len(test_methods) == len(BINARY_TRANSFORMS)

if __name__ == "__main__":
  test_lib.main()
