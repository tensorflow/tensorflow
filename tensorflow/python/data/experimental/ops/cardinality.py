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
"""Cardinality analysis of `Dataset` objects."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export


INFINITE = -1
UNKNOWN = -2
tf_export("data.experimental.INFINITE_CARDINALITY").export_constant(
    __name__, "INFINITE")
tf_export("data.experimental.UNKNOWN_CARDINALITY").export_constant(
    __name__, "UNKNOWN")


@tf_export("data.experimental.cardinality")
def cardinality(dataset):
  """Returns the cardinality of `dataset`, if known.

  The operation returns the cardinality of `dataset`. The operation may return
  `tf.data.experimental.INFINITE_CARDINALITY` if `dataset` contains an infinite
  number of elements or `tf.data.experimental.UNKNOWN_CARDINALITY` if the
  analysis fails to determine the number of elements in `dataset` (e.g. when the
  dataset source is a file).

  Args:
    dataset: A `tf.data.Dataset` for which to determine cardinality.

  Returns:
    A scalar `tf.int64` `Tensor` representing the cardinality of `dataset`. If
    the cardinality is infinite or unknown, the operation returns the named
    constant `INFINITE_CARDINALITY` and `UNKNOWN_CARDINALITY` respectively.
  """
  return ged_ops.experimental_dataset_cardinality(dataset._as_variant_tensor())  # pylint: disable=protected-access
