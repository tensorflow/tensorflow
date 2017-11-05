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
"""Library for benchmarking OpKernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import time

from tensorflow.python.framework import ops


def device(use_gpu=False):
  """TensorFlow device to assign ops to."""
  if use_gpu:
    return ops.device("/gpu:0")
  return ops.device("/cpu:0")


def seconds_per_run(op, sess, num_runs=50):
  """Number of seconds taken to execute 'op' once on average."""
  for _ in range(2):
    sess.run(op)

  start_time = time.time()
  for _ in range(num_runs):
    sess.run(op)

  end_time = time.time()
  time_taken = (end_time - start_time) / num_runs
  return time_taken


def dict_product(dicts):
  """Constructs iterator over outer product of entries in a dict-of-lists.

  Example:
    >>> dict_products({"a": [1,2], "b": [3, 4]})
    >>> [{"a": 1, "b": 3},
         {"a": 1, "b": 4},
         {"a": 2, "b": 3},
         {"a": 2, "b": 4}]

  Args:
    dicts: dictionary with string keys and list values.

  Yields:
    Individual dicts from outer product.
  """
  keys, values = zip(*dicts.items())
  for config_values in itertools.product(*values):
    yield dict(zip(keys, config_values))
