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
"""Resampling dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import resampling
from tensorflow.python.util import deprecation


@deprecation.deprecated(None,
                        "Use `tf.data.experimental.rejection_resample(...)`.")
def rejection_resample(class_func, target_dist, initial_dist=None, seed=None):
  """A transformation that resamples a dataset to achieve a target distribution.

  **NOTE** Resampling is performed via rejection sampling; some fraction
  of the input values will be dropped.

  Args:
    class_func: A function mapping an element of the input dataset to a scalar
      `tf.int32` tensor. Values should be in `[0, num_classes)`.
    target_dist: A floating point type tensor, shaped `[num_classes]`.
    initial_dist: (Optional.)  A floating point type tensor, shaped
      `[num_classes]`.  If not provided, the true class distribution is
      estimated live in a streaming fashion.
    seed: (Optional.) Python integer seed for the resampler.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  return resampling.rejection_resample(class_func, target_dist, initial_dist,
                                       seed)
