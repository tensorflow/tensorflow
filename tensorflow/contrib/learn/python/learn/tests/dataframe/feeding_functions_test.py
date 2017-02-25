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
"""Tests feeding functions using arrays and `DataFrames`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.learn.python.learn.dataframe.queues import feeding_functions as ff
from tensorflow.python.platform import test

# pylint: disable=g-import-not-at-top
try:
  import pandas as pd
  HAS_PANDAS = True
except ImportError:
  HAS_PANDAS = False


def vals_to_list(a):
  return {
      key: val.tolist() if isinstance(val, np.ndarray) else val
      for key, val in a.items()
  }


class _FeedingFunctionsTestCase(test.TestCase):
  """Tests for feeding functions."""

  def testArrayFeedFnBatchOne(self):
    array = np.arange(32).reshape([16, 2])
    placeholders = ["index_placeholder", "value_placeholder"]
    aff = ff._ArrayFeedFn(placeholders, array, 1)

    # cycle around a couple times
    for x in range(0, 100):
      i = x % 16
      expected = {
          "index_placeholder": [i],
          "value_placeholder": [[2 * i, 2 * i + 1]]
      }
      actual = aff()
      self.assertEqual(expected, vals_to_list(actual))

  def testArrayFeedFnBatchFive(self):
    array = np.arange(32).reshape([16, 2])
    placeholders = ["index_placeholder", "value_placeholder"]
    aff = ff._ArrayFeedFn(placeholders, array, 5)

    # cycle around a couple times
    for _ in range(0, 101, 2):
      aff()

    expected = {
        "index_placeholder": [15, 0, 1, 2, 3],
        "value_placeholder": [[30, 31], [0, 1], [2, 3], [4, 5], [6, 7]]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testArrayFeedFnBatchTwoWithOneEpoch(self):
    array = np.arange(5) + 10
    placeholders = ["index_placeholder", "value_placeholder"]
    aff = ff._ArrayFeedFn(placeholders, array, batch_size=2, num_epochs=1)

    expected = {
        "index_placeholder": [0, 1],
        "value_placeholder": [10, 11]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

    expected = {
        "index_placeholder": [2, 3],
        "value_placeholder": [12, 13]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

    expected = {
        "index_placeholder": [4],
        "value_placeholder": [14]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testArrayFeedFnBatchOneHundred(self):
    array = np.arange(32).reshape([16, 2])
    placeholders = ["index_placeholder", "value_placeholder"]
    aff = ff._ArrayFeedFn(placeholders, array, 100)

    expected = {
        "index_placeholder":
            list(range(0, 16)) * 6 + list(range(0, 4)),
        "value_placeholder":
            np.arange(32).reshape([16, 2]).tolist() * 6 +
            [[0, 1], [2, 3], [4, 5], [6, 7]]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testArrayFeedFnBatchOneHundredWithSmallerArrayAndMultipleEpochs(self):
    array = np.arange(2) + 10
    placeholders = ["index_placeholder", "value_placeholder"]
    aff = ff._ArrayFeedFn(placeholders, array, batch_size=100, num_epochs=2)

    expected = {
        "index_placeholder": [0, 1, 0, 1],
        "value_placeholder": [10, 11, 10, 11],
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testPandasFeedFnBatchOne(self):
    if not HAS_PANDAS:
      return
    array1 = np.arange(32, 64)
    array2 = np.arange(64, 96)
    df = pd.DataFrame({"a": array1, "b": array2}, index=np.arange(96, 128))
    placeholders = ["index_placeholder", "a_placeholder", "b_placeholder"]
    aff = ff._PandasFeedFn(placeholders, df, 1)

    # cycle around a couple times
    for x in range(0, 100):
      i = x % 32
      expected = {
          "index_placeholder": [i + 96],
          "a_placeholder": [32 + i],
          "b_placeholder": [64 + i]
      }
      actual = aff()
      self.assertEqual(expected, vals_to_list(actual))

  def testPandasFeedFnBatchFive(self):
    if not HAS_PANDAS:
      return
    array1 = np.arange(32, 64)
    array2 = np.arange(64, 96)
    df = pd.DataFrame({"a": array1, "b": array2}, index=np.arange(96, 128))
    placeholders = ["index_placeholder", "a_placeholder", "b_placeholder"]
    aff = ff._PandasFeedFn(placeholders, df, 5)

    # cycle around a couple times
    for _ in range(0, 101, 2):
      aff()

    expected = {
        "index_placeholder": [127, 96, 97, 98, 99],
        "a_placeholder": [63, 32, 33, 34, 35],
        "b_placeholder": [95, 64, 65, 66, 67]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testPandasFeedFnBatchTwoWithOneEpoch(self):
    if not HAS_PANDAS:
      return
    array1 = np.arange(32, 37)
    array2 = np.arange(64, 69)
    df = pd.DataFrame({"a": array1, "b": array2}, index=np.arange(96, 101))
    placeholders = ["index_placeholder", "a_placeholder", "b_placeholder"]
    aff = ff._PandasFeedFn(placeholders, df, batch_size=2, num_epochs=1)

    expected = {
        "index_placeholder": [96, 97],
        "a_placeholder": [32, 33],
        "b_placeholder": [64, 65]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

    expected = {
        "index_placeholder": [98, 99],
        "a_placeholder": [34, 35],
        "b_placeholder": [66, 67]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

    expected = {
        "index_placeholder": [100],
        "a_placeholder": [36],
        "b_placeholder": [68]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testPandasFeedFnBatchOneHundred(self):
    if not HAS_PANDAS:
      return
    array1 = np.arange(32, 64)
    array2 = np.arange(64, 96)
    df = pd.DataFrame({"a": array1, "b": array2}, index=np.arange(96, 128))
    placeholders = ["index_placeholder", "a_placeholder", "b_placeholder"]
    aff = ff._PandasFeedFn(placeholders, df, 100)

    expected = {
        "index_placeholder": list(range(96, 128)) * 3 + list(range(96, 100)),
        "a_placeholder": list(range(32, 64)) * 3 + list(range(32, 36)),
        "b_placeholder": list(range(64, 96)) * 3 + list(range(64, 68))
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testPandasFeedFnBatchOneHundredWithSmallDataArrayAndMultipleEpochs(self):
    if not HAS_PANDAS:
      return
    array1 = np.arange(32, 34)
    array2 = np.arange(64, 66)
    df = pd.DataFrame({"a": array1, "b": array2}, index=np.arange(96, 98))
    placeholders = ["index_placeholder", "a_placeholder", "b_placeholder"]
    aff = ff._PandasFeedFn(placeholders, df, batch_size=100, num_epochs=2)

    expected = {
        "index_placeholder": [96, 97, 96, 97],
        "a_placeholder": [32, 33, 32, 33],
        "b_placeholder": [64, 65, 64, 65]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testOrderedDictNumpyFeedFnBatchTwoWithOneEpoch(self):
    a = np.arange(32, 37)
    b = np.arange(64, 69)
    x = {"a": a, "b": b}
    ordered_dict_x = collections.OrderedDict(
        sorted(x.items(), key=lambda t: t[0]))
    placeholders = ["index_placeholder", "a_placeholder", "b_placeholder"]
    aff = ff._OrderedDictNumpyFeedFn(
        placeholders, ordered_dict_x, batch_size=2, num_epochs=1)

    expected = {
        "index_placeholder": [0, 1],
        "a_placeholder": [32, 33],
        "b_placeholder": [64, 65]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

    expected = {
        "index_placeholder": [2, 3],
        "a_placeholder": [34, 35],
        "b_placeholder": [66, 67]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

    expected = {
        "index_placeholder": [4],
        "a_placeholder": [36],
        "b_placeholder": [68]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testOrderedDictNumpyFeedFnLargeBatchWithSmallArrayAndMultipleEpochs(self):
    a = np.arange(32, 34)
    b = np.arange(64, 66)
    x = {"a": a, "b": b}
    ordered_dict_x = collections.OrderedDict(
        sorted(x.items(), key=lambda t: t[0]))
    placeholders = ["index_placeholder", "a_placeholder", "b_placeholder"]
    aff = ff._OrderedDictNumpyFeedFn(
        placeholders, ordered_dict_x, batch_size=100, num_epochs=2)

    expected = {
        "index_placeholder": [0, 1, 0, 1],
        "a_placeholder": [32, 33, 32, 33],
        "b_placeholder": [64, 65, 64, 65]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))


if __name__ == "__main__":
  test.main()
