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
"""Test data utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_quantile_based_buckets(feature_values, num_buckets):
  quantiles = np.percentile(
      np.array(feature_values), ([100 * (i + 1.) / (num_buckets + 1.)
                                  for i in range(num_buckets)]))
  return list(quantiles)


def prepare_iris_data_for_logistic_regression():
  # Converts iris data to a logistic regression problem.
  iris = tf.contrib.learn.datasets.load_iris()
  ids = np.where((iris.target == 0) | (iris.target == 1))
  return tf.contrib.learn.datasets.base.Dataset(data=iris.data[ids],
                                                target=iris.target[ids])


def iris_input_multiclass_fn():
  iris = tf.contrib.learn.datasets.load_iris()
  return {
      'feature': tf.constant(iris.data, dtype=tf.float32)
  }, tf.constant(iris.target, shape=(150, 1), dtype=tf.int32)


def iris_input_logistic_fn():
  iris = prepare_iris_data_for_logistic_regression()
  return {
      'feature': tf.constant(iris.data, dtype=tf.float32)
  }, tf.constant(iris.target, shape=(100, 1), dtype=tf.int32)
