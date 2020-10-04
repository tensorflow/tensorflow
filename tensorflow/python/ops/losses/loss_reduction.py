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
"""The TF2 version of the enum keras.losses.Reduction."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ReductionV2(object):
  """Types of loss reduction.

  Contains the following values:

  * `AUTO`: Indicates that the reduction option will be determined by the usage
     context. For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When
     used with `tf.distribute.Strategy`, outside of built-in training loops such
     as `tf.keras` `compile` and `fit`, we expect reduction value to be
     `SUM` or `NONE`. Using `AUTO` in that case will raise an error.
  * `NONE`: Weighted losses with one dimension reduced (axis=-1, or axis
     specified by loss function). When this reduction type used with built-in
     Keras training loops like `fit`/`evaluate`, the unreduced vector loss is
     passed to the optimizer but the reported loss will be a scalar value.
  * `SUM`: Scalar sum of weighted losses.
  * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
     This reduction type is not supported when used with
     `tf.distribute.Strategy` outside of built-in training loops like `tf.keras`
     `compile`/`fit`.

     You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
     ```
     with strategy.scope():
       loss_obj = tf.keras.losses.CategoricalCrossentropy(
           reduction=tf.keras.losses.Reduction.NONE)
       ....
       loss = tf.reduce_sum(loss_obj(labels, predictions)) *
           (1. / global_batch_size)
     ```

  Please see the
  [custom training guide](https://www.tensorflow.org/tutorials/distribute/custom_training)  # pylint: disable=line-too-long
  for more details on this.
  """

  AUTO = 'auto'
  NONE = 'none'
  SUM = 'sum'
  SUM_OVER_BATCH_SIZE = 'sum_over_batch_size'

  @classmethod
  def all(cls):
    return (cls.AUTO, cls.NONE, cls.SUM, cls.SUM_OVER_BATCH_SIZE)

  @classmethod
  def validate(cls, key):
    if key not in cls.all():
      raise ValueError('Invalid Reduction Key %s.' % key)
