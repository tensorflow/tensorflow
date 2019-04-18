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

  * `NONE`: Un-reduced weighted losses with the same shape as input.
  * `SUM`: Scalar sum of weighted losses.
  * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
     Note that when using `tf.distribute.Strategy`, this is just the
     replica-local batch size.
  """

  NONE = 'none'
  SUM = 'sum'
  SUM_OVER_BATCH_SIZE = 'sum_over_batch_size'

  @classmethod
  def all(cls):
    return (cls.NONE, cls.SUM, cls.SUM_OVER_BATCH_SIZE)

  @classmethod
  def validate(cls, key):
    if key not in cls.all():
      raise ValueError('Invalid Reduction Key %s.' % key)
