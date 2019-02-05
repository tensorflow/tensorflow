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
"""Benchmarks for `tf.data.Dataset.from_tensor_slices()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


# TODO(b/119837791): Add eager benchmarks.
class FromTensorSlicesBenchmark(test.Benchmark):
  """Benchmarks for `tf.data.Dataset.from_tensor_slices()`."""

  def benchmarkSliceRepeatBatch(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data)
        .repeat(num_epochs + 1).batch(batch_size))
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    with session.Session() as sess:
      sess.run(iterator.initializer)
      # Run one whole epoch to burn in the computation.
      for _ in range(input_size // batch_size):
        sess.run(next_element)
      deltas = []
      try:
        while True:
          start = time.time()
          sess.run(next_element)
          deltas.append(time.time() - start)
      except errors.OutOfRangeError:
        pass

    median_wall_time = np.median(deltas)
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name="slice_repeat_batch_input_%d_batch_%d" % (input_size, batch_size))

  def benchmarkSliceRepeatBatchCallable(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data)
        .repeat(num_epochs + 1).batch(batch_size))
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    with session.Session() as sess:
      sess.run(iterator.initializer)
      get_next_element = sess.make_callable(next_element)
      # Run one whole epoch to burn in the computation.
      for _ in range(input_size // batch_size):
        get_next_element()
      deltas = []
      try:
        while True:
          start = time.time()
          get_next_element()
          deltas.append(time.time() - start)
      except errors.OutOfRangeError:
        pass

    median_wall_time = np.median(deltas)
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name="slice_repeat_batch_callable_input_%d_batch_%d" %
        (input_size, batch_size))

  def benchmarkReshapeSliceRepeatCallable(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data.reshape(100, 100))
        .repeat(num_epochs + 1))
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    with session.Session() as sess:
      sess.run(iterator.initializer)
      get_next_element = sess.make_callable(next_element)
      # Run one whole epoch to burn in the computation.
      for _ in range(input_size // batch_size):
        get_next_element()
      deltas = []
      try:
        while True:
          start = time.time()
          get_next_element()
          deltas.append(time.time() - start)
      except errors.OutOfRangeError:
        pass

    median_wall_time = np.median(deltas)
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name="reshape_slice_repeat_callable_input_%d_batch_%d" %
        (input_size, batch_size))

  def benchmarkSliceBatchCacheRepeatCallable(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data).batch(batch_size)
        .cache().repeat(num_epochs + 1))
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    with session.Session() as sess:
      sess.run(iterator.initializer)
      get_next_element = sess.make_callable(next_element)
      # Run one whole epoch to burn in the computation.
      for _ in range(input_size // batch_size):
        get_next_element()
      deltas = []
      try:
        while True:
          start = time.time()
          get_next_element()
          deltas.append(time.time() - start)
      except errors.OutOfRangeError:
        pass

    median_wall_time = np.median(deltas)
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name="slice_batch_cache_repeat_callable_input_%d_batch_%d" %
        (input_size, batch_size))


if __name__ == "__main__":
  test.main()
