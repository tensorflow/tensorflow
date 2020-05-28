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
"""Utilities for reduce operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

from tensorflow.python.ops import variable_scope
from tensorflow.python.util.tf_export import tf_export


@tf_export("distribute.ReduceOp")
class ReduceOp(enum.Enum):
  """Indicates how a set of values should be reduced.

  * `SUM`: Add all the values.
  * `MEAN`: Take the arithmetic mean ("average") of the values.
  """
  # TODO(priyag): Add the following types:
  # `MIN`: Return the minimum of all values.
  # `MAX`: Return the maximum of all values.
  SUM = "SUM"
  MEAN = "MEAN"

  @staticmethod
  def from_variable_aggregation(aggregation):
    mapping = {
        variable_scope.VariableAggregation.SUM: ReduceOp.SUM,
        variable_scope.VariableAggregation.MEAN: ReduceOp.MEAN,
    }

    reduce_op = mapping.get(aggregation)
    if not reduce_op:
      raise ValueError("Could not convert from `tf.VariableAggregation` %s to"
                       "`tf.distribute.ReduceOp` type" % aggregation)
    return reduce_op
