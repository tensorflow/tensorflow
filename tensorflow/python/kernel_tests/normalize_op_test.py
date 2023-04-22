# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.norm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_impl
from tensorflow.python.platform import test as test_lib


def _AddTest(test, test_name, fn):
  test_name = "_".join(["test", test_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


# pylint: disable=redefined-builtin
def _Normalize(x, ord, axis):
  if isinstance(axis, (list, tuple)):
    norm = np.linalg.norm(x, ord, tuple(axis))
    if axis[0] < axis[1]:
      # This prevents axis to be inserted in-between
      # e.g. when (-2, -1)
      for d in reversed(axis):
        norm = np.expand_dims(norm, d)
    else:
      for d in axis:
        norm = np.expand_dims(norm, d)
    return x / norm
  elif axis is None:
    # Tensorflow handles None differently
    norm = np.linalg.norm(x.flatten(), ord, axis)
    return x / norm
  else:
    norm = np.apply_along_axis(np.linalg.norm, axis, x, ord)
    return x / np.expand_dims(norm, axis)


class NormalizeOpTest(test_lib.TestCase):
  pass


def _GetNormalizeOpTest(dtype_, shape_, ord_, axis_):

  @test_util.run_in_graph_and_eager_modes
  def Test(self):
    is_matrix_norm = (isinstance(axis_, tuple) or
                      isinstance(axis_, list)) and len(axis_) == 2
    is_fancy_p_norm = np.isreal(ord_) and np.floor(ord_) != ord_
    if ((not is_matrix_norm and ord_ == "fro") or
        (is_matrix_norm and is_fancy_p_norm)):
      self.skipTest("Not supported by neither numpy.linalg.norm nor tf.norm")
    if ord_ == "euclidean" or (axis_ is None and len(shape) > 2):
      self.skipTest("Not supported by numpy.linalg.norm")
    matrix = np.random.randn(*shape_).astype(dtype_)
    if dtype_ in (np.complex64, np.complex128):
      matrix += 1j * np.random.randn(*shape_).astype(dtype_)
    tf_np_n, _ = self.evaluate(nn_impl.normalize(matrix, ord_, axis_))
    np_n = _Normalize(matrix, ord_, axis_)
    self.assertAllClose(tf_np_n, np_n, rtol=1e-5, atol=1e-5)

  return Test


# pylint: disable=redefined-builtin
if __name__ == "__main__":
  for dtype in np.float32, np.float64, np.complex64, np.complex128:
    for rows in 2, 5:
      for cols in 2, 5:
        for batch in [], [2], [2, 3]:
          shape = batch + [rows, cols]
          for ord in "euclidean", "fro", 0.5, 1, 2, np.inf:
            for axis in [
                None, (-2, -1), (-1, -2), -len(shape), 0,
                len(shape) - 1
            ]:
              name = "%s_%s_ord_%s_axis_%s" % (dtype.__name__, "_".join(
                  map(str, shape)), ord, axis)
              _AddTest(NormalizeOpTest, "Normalize_" + name,
                       _GetNormalizeOpTest(dtype, shape, ord, axis))

  test_lib.main()
