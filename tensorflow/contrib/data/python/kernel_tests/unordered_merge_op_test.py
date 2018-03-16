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

from tensorflow.python.data.ops import dataset_ops
from tensorflow.contrib.data.python.ops import unordered_merge, grouping
from tensorflow.python.framework import dtypes, errors, tensor_shape
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.platform import test


class UnorderedMergeDatasetTest(test.TestCase): #pylint: disable=missing-docstring
  def testUnorderedMergeDatasetErrorArgument(self): #pylint: disable=missing-docstring
    error_arg_ok = False
    try:
      _ = unordered_merge.unordered_merge([1, 2, 3])
    except TypeError as e:
      error_arg_ok = True
      if e.args[0].find('argument'):
        error_arg_ok = True
    self.assertEqual(error_arg_ok, True)

  def testUnorderedMergeDatasetErrorDtype(self): #pylint: disable=missing-docstring
    types = [dtypes.int64, dtypes.float32]
    placeholders = [array_ops.placeholder(t, shape=[None])
                    for t in types]
    datasets = [dataset_ops.Dataset.from_tensor_slices(p)
                for p in placeholders]
    datasets = [ds.repeat(5).shuffle(10) for ds in datasets]

    new_datasets = []
    for i in range(3):
      new_datasets.append(datasets[0].shard(4, i).apply(
          grouping.group_by_window(
              lambda x: x % 2, lambda _, xs: xs.batch(4), 4)))
    new_datasets.append(datasets[1].batch(4))

    # for checking messages
    error_types_ok = False
    try:
      _ = unordered_merge.unordered_merge(new_datasets)
    except TypeError as e:
      if e.args[0].find('types'):
        error_types_ok = True
    self.assertEqual(error_types_ok, True)

  def testUnorderedMergeDatasetErrorShape(self): #pylint: disable=missing-docstring
    placeholders = [array_ops.placeholder(dtypes.int64, shape=[None])
                    for _ in range(3)]

    datasets = [dataset_ops.Dataset.from_tensor_slices(p)
                for p in placeholders]
    datasets = [ds.repeat(5).shuffle(10) for ds in datasets]

    new_datasets = []
    for i in range(3):
      new_datasets.append(datasets[0].shard(4, i).apply(
          grouping.group_by_window(lambda x: x%2, lambda _, xs: xs.batch(4), 4)))
    new_datasets.append(datasets[1])

    error_shapes_ok = False
    try:
      _ = unordered_merge.unordered_merge(new_datasets)
    except TypeError as e:
      if e.args[0].find('shapes'):
        error_shapes_ok = True
    self.assertEqual(error_shapes_ok, True)

    new_datasets = []
    for i in range(3):
      new_datasets.append(datasets[0].shard(4, i).apply(
          grouping.group_by_window(lambda x: x%2, lambda _, xs: xs.batch(4), 4)))
    ds2 = dataset_ops.Dataset.zip(tuple([datasets[1], datasets[2]]))
    new_datasets.append(ds2.shard(4, 3).apply(
        grouping.group_by_window(lambda x, y: x % 2, lambda _, xs: xs.batch(4), 4)))

    error_shapes_ok2 = False
    try:
      _ = unordered_merge.unordered_merge(new_datasets)
    except TypeError as e:
      if e.args[0].find('shapes'):
        error_shapes_ok2 = True
    self.assertEqual(error_shapes_ok2, True)

  def testUnorderedMergeDatasetNormalSimple(self): #pylint: disable=missing-docstring
    data_num = 1000

    dataset = dataset_ops.Dataset.range(data_num)
    datasets = [dataset.shard(10, i).shuffle(50) for i in range(10)]
    dataset = unordered_merge.unordered_merge(datasets)
    next_one_op = dataset.make_one_shot_iterator().get_next()

    results = []
    with self.test_session() as session:
      try:
        while True:
          results.append(session.run(next_one_op))
      except errors.OutOfRangeError:
        pass
    sorted_result = sorted(results)
    expectation = list(range(data_num))

    self.assertEqual(sorted_result, expectation)

  def testUnorderedMergeDatasetNormalDictionary(self): #pylint: disable=missing-docstring
    data_lengths = [4, 6, 9, 12, 18]
    lcm = 36
    data_num = 300

    def gen():
      for i in range(data_num):
        for j in data_lengths:
          if i%j == 0:
            yield [j] + [i+k for k in range(j)]

    def code(tensor):
      return {'key': tensor[0],
              'value': array_ops.slice(tensor, [1], [-1])}

    def key_fn(data):
      return data['key']

    def window_size_fn(key):
      return math_ops.cast(lcm/key, dtypes.int64)

    def reduce_fn(key, ds):
      return ds.batch(window_size_fn(key))

    dataset = dataset_ops.Dataset.from_generator(
        gen, dtypes.int64, tensor_shape.TensorShape([None])).map(code)
    datasets = [dataset.shard(10, i) for i in range(10)]
    datasets = [ds.apply(grouping.group_by_window(
        key_fn, reduce_fn, window_size_func=window_size_fn))
                for ds in datasets]
    dataset = unordered_merge.unordered_merge(datasets)
    next_one_op = dataset.make_one_shot_iterator().get_next()

    result = dict()
    for j in data_lengths:
      result[j] = []
    with self.test_session() as session:
      try:
        while True:
          got = session.run(next_one_op)
          for (i, key) in enumerate(got['key']):
            result[key] += got['value'][i].tolist()
      except errors.OutOfRangeError:
        pass

    for j in data_lengths:
      rem = data_num%j
      n = data_num if rem == 0 else data_num+j-rem
      expectation = list(range(n))
      self.assertEqual(expectation, sorted(result[j]))


if __name__ == "__main__":
  test.main()
