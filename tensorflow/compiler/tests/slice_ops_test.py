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
"""Tests for slicing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest



class SliceTest(XLATestCase):

  def test1D(self):
    for dtype in self.numeric_types:
      with self.test_session():
        i = array_ops.placeholder(dtype, shape=[10])
        with self.test_scope():
          o = array_ops.slice(i, [2], [4])
        params = {
            i: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([2, 3, 4, 5], result)

  def test3D(self):
    for dtype in self.numeric_types:
      with self.test_session():
        i = array_ops.placeholder(dtype, shape=[3, 3, 10])
        with self.test_scope():
          o = array_ops.slice(i, [1, 2, 2], [1, 1, 4])
        params = {
            i: [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                 [5, 3, 1, 7, 9, 2, 4, 6, 8, 0]],
                [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [8, 7, 6, 5, 4, 3, 2, 1, 8, 7]],
                [[7, 5, 7, 5, 7, 5, 7, 5, 7, 5],
                 [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                 [9, 8, 7, 9, 8, 7, 9, 8, 7, 9]]]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[[6, 5, 4, 3]]], result)



class StridedSliceTest(XLATestCase):

  def test1D(self):
    for dtype in self.numeric_types:
      with self.test_session():
        i = array_ops.placeholder(dtype, shape=[10])
        with self.test_scope():
          o = array_ops.strided_slice(i, [2], [6], [2])
        params = {
            i: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([2, 4], result)

  def test1DNegtiveStride(self):
    for dtype in self.numeric_types:
      with self.test_session():
        i = array_ops.placeholder(dtype, shape=[10])
        with self.test_scope():
          o = array_ops.strided_slice(i, [6], [2], [-2])
        params = {
            i: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([6, 4], result)

  def test3D(self):
    for dtype in self.numeric_types:
      with self.test_session():
        i = array_ops.placeholder(dtype, shape=[3, 3, 10])
        with self.test_scope():
          o = array_ops.strided_slice(i, [0, 2, 2], [2, 3, 6], [1, 1, 2])
        params = {
            i: [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                 [5, 3, 1, 7, 9, 2, 4, 6, 8, 0]],
                [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [8, 7, 6, 5, 4, 3, 2, 1, 8, 7]],
                [[7, 5, 7, 5, 7, 5, 7, 5, 7, 5],
                 [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                 [9, 8, 7, 9, 8, 7, 9, 8, 7, 9]]]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[[1, 9]], [[6, 4]]], result)

  def test3DNegativeStride(self):
    for dtype in self.numeric_types:
      with self.test_session():
        i = array_ops.placeholder(dtype, shape=[3, 4, 10])
        with self.test_scope():
          o = array_ops.strided_slice(i, [2, 2, 6], [0, 0, 2], [-1, -1, -2])
        params = {
            i: [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                 [5, 3, 1, 7, 9, 2, 4, 6, 8, 0],
                 [4, 5, 2, 4, 3, 7, 6, 8, 9, 4]],
                [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 [4, 3, 4, 5, 7, 6, 5, 3, 4, 5],
                 [8, 7, 6, 5, 4, 3, 2, 1, 8, 7],
                 [7, 1, 7, 1, 8, 1, 8, 1, 3, 1]],
                [[7, 5, 7, 5, 7, 5, 7, 5, 7, 5],
                 [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                 [9, 8, 7, 9, 8, 7, 9, 8, 7, 9],
                 [9, 9, 5, 5, 6, 6, 3, 3, 6, 6]]]
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[[9, 8],
                              [1, 1]],
                             [[2, 4],
                              [5, 7]]], result)

if __name__ == "__main__":
  googletest.main()
