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
"""Tests for connected component analysis."""


import numpy as np

from tensorflow.contrib.image.python.ops import image_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class RecursiveFillTest(test_util.TensorFlowTestCase):
  """Set of test to try the recursive fill function.
  """

  def test_simple_2d(self):
    """Simple 2D test to visualize what the algorithm is doing.
    """
    orig = math_ops.cast(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]],
				    dtypes.float32)
    expected = (
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    with self.test_session():
      result = image_ops.recursive_fill(orig, 4, 2)
      variables.global_variables_initializer().run()
      self.assertAllEqual(result.eval(), expected)

  def test_random_2d_fills(self):
    """ Several random 2D executions with different thresholds
    """
    for _ in range(10):
      for mat_thr in range(3, 8):
        with self.test_session():
          #mat_thr = 4
          init = 1.0*(np.random.random((40, 40)) > 0.8)
          init_tensor = math_ops.cast(init, dtypes.float32)

          expected = seq_fill_2d(init, mat_thr)
          result = image_ops.recursive_fill(init_tensor, mat_thr, 2)
          variables.global_variables_initializer().run()
          self.assertAllEqual(result.eval(), expected)

  def test_random_3d_fills(self):
    """Several random 3D executions with different thresholds.
    """
    for _ in range(10):
      for mat_thr in range(12, 20):
        with self.test_session():
          #mat_thr = 4
          init = 1.0*(np.random.random((10, 10, 10)) > 0.8)
          init_tensor = math_ops.cast(init, dtypes.float32)

          expected = seq_fill_3d(init, mat_thr)
          result = image_ops.recursive_fill(init_tensor, mat_thr, 3)
          variables.global_variables_initializer().run()
          self.assertAllEqual(result.eval(), expected)



def seq_fill_2d(arr, thr):
  """Iterative function (2D) used to confirm the TF implementation works.
  """
  again = True
  c = 0
  while again:
    c += 1
    again = False
    ind = np.zeros_like(arr)
    for i in range(arr.shape[0]):
      for j in range(arr.shape[1]):
        if arr[i, j] == 0 and np.sum(
            arr[np.clip(i-1, 0, 999):i+2, np.clip(j-1, 0, 999):j+2]) >= thr:
          ind[i, j] = 1
          again = True
    arr += ind
  return arr

def seq_fill_3d(arr, thr):
  """Iterative function (3D) used to confirm the TF implementation works.
  """
  again = True
  c = 0
  while again:
    c += 1
    again = False
    ind = np.zeros_like(arr)
    for i in range(arr.shape[0]):
      for j in range(arr.shape[1]):
        for k in range(arr.shape[2]):
          if arr[i, j, k] == 0 and np.sum(
              arr[np.clip(i-1, 0, 999):i+2, np.clip(j-1, 0, 999):j+2,
                  np.clip(k-1, 0, 999):k+2]) >= thr:
            ind[i, j, k] = 1
            again = True
    arr += ind
  return arr

if __name__ == '__main__':
  googletest.main()
