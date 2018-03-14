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

from tensorflow.python.data.ops import dataset_ops
from tensorflow.contrib.data.python.ops import unordered_merge, grouping
from tensorflow.python.framework import dtypes, errors
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.platform import test


class UnorderedMergeDatasetTest(test.TestCase):
  def testUnorderedMergeDatasetErrorArgument(self):
    error_arg_ok = False
    try:
      dataset = unordered_merge.unordered_merge([1, 2, 3])
    except TypeError as e:
        error_arg_ok = True
        if(e.args[0].find('argument')):
          error_arg_ok = True
    except Exception:
      pass
    else:
      pass
    self.assertEqual(error_arg_ok, True)

  def testUnorderedMergeDatasetErrorDtype(self):
    #TODO: rearrange
    #placeholders = [
    #    array_ops.placeholder(dtypes.int64),
    #    array_ops.placeholder(dtypes.float32)
    #]

    #datasets = [dataset_ops.Dataset.from_tensor_slices(p)
    #            for p in placeholders]
    #datasets = [ds.repeat(5).shuffle(10) for ds in datasets]
    #new_datasets = []
    #for i in range(3):
    #  new_datasets.append(datasets[0].shard(4, i).apply(
    #      grouping.group_by_window(lambda x: x%2, lambda _, xs: xs.batch(4), 4)))
    #new_datasets.append(datasets[1].batch(4))
    ds1 = dataset_ops.Dataset.range(5).repeat(5).shuffle(10)
    ds2 = dataset_ops.Dataset.range(5).map(
            lambda x: math_ops.cast(x, dtypes.float32)).shuffle(5)
    new_datasets = []
    for i in range(3):
      new_datasets.append(ds1.shard(4, i).apply(
          grouping.group_by_window(
              lambda x: x % 2, lambda _, xs: xs.batch(4), 4)))
    new_datasets.append(ds2.batch(4))

    # for checking messages
    error_types_ok = False
    try:
      dataset = unordered_merge.unordered_merge(new_datasets)
    except TypeError as e:
        if(e.args[0].find('types')):
          error_types_ok = True
    except Exception as e:
      self.assertEqual(e, True)
      pass
    self.assertEqual(error_types_ok, True)

  def testUnorderedMergeDatasetErrorShape(self):
    #TODO: rearrange
    placeholders = [
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.int64)
    ]

    #datasets = [dataset_ops.Dataset.from_tensor_slices(p)
    #            for p in placeholders]
    #datasets = [ds.repeat(5).shuffle(10) for ds in datasets]
    
    #new_datasets = []
    #for i in range(3):
    #  new_datasets.append(datasets[0].shard(4, i).apply(
    #    grouping.group_by_window(lambda x: x%2, lambda _, xs: xs.batch(4), 4)))
    #new_datasets.append(datasets[1])

     
    ds1 = dataset_ops.Dataset.range(5).repeat(5).shuffle(10)
    ds2 = dataset_ops.Dataset.range(5).repeat(5).shuffle(10)
    datasets = []
    for i in range(3):
      datasets.append(ds1.shard(4, i).apply(
          grouping.group_by_window(
              lambda x: x % 2, lambda _, xs: xs.batch(4), 4)))
    datasets.append(ds2)

    error_shapes_ok = False
    try:
      dataset = unordered_merge.unordered_merge(datasets)
    except TypeError as e:
        if(e.args[0].find('shapes')):
          error_shapes_ok = True
    except Exception:
      pass
    self.assertEqual(error_shapes_ok, True)

    new_datasets = []
    for i in range(3):
      new_datasets.append(ds1.shard(4, i).apply(
        grouping.group_by_window(lambda x: x%2, lambda _, xs: xs.batch(4), 4)))
    dsX = dataset_ops.Dataset.zip(tuple([ds1, ds2]))
    new_datasets.append(dsX.shard(4, 3).apply(
        grouping.group_by_window(lambda x, y: x % 2, lambda _, xs: xs.batch(4), 4)))

    error_shapes_ok = False
    try:
      dataset = unordered_merge.unordered_merge(new_datasets)
    except TypeError as e:
        if(e.args[0].find('shapes')):
          error_shapes_ok = True
    except Exception:
      pass
    self.assertEqual(error_shapes_ok, True)

  #TODO: add normal cases

if __name__ == "__main__":
  test.main()
