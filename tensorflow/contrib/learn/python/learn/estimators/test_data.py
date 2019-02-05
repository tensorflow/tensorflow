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
"""Test data utilities (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes


def get_quantile_based_buckets(feature_values, num_buckets):
  quantiles = np.percentile(
      np.array(feature_values),
      ([100 * (i + 1.) / (num_buckets + 1.) for i in range(num_buckets)]))
  return list(quantiles)


def prepare_iris_data_for_logistic_regression():
  # Converts iris data to a logistic regression problem.
  iris = base.load_iris()
  ids = np.where((iris.target == 0) | (iris.target == 1))
  return base.Dataset(data=iris.data[ids], target=iris.target[ids])


def iris_input_multiclass_fn():
  iris = base.load_iris()
  return {
      'feature': constant_op.constant(
          iris.data, dtype=dtypes.float32)
  }, constant_op.constant(
      iris.target, shape=(150, 1), dtype=dtypes.int32)


def iris_input_logistic_fn():
  iris = prepare_iris_data_for_logistic_regression()
  return {
      'feature': constant_op.constant(
          iris.data, dtype=dtypes.float32)
  }, constant_op.constant(
      iris.target, shape=(100, 1), dtype=dtypes.int32)
