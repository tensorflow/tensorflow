
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

"""Dropout operations and handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.layers import dropout as contrib_dropout

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs


# Key to collect dropout probabilities.
DROPOUTS = "dropouts"


@deprecated(date="2016-08-15",
            instructions="Please use tf.contrib.layers.dropout instead.")
def dropout(tensor_in, prob, name=None):
  """Adds dropout node and stores probability tensor into graph collection.

  This is deprecated. Please use contrib.layers.dropout instead.

  Args:
    tensor_in: Input tensor.
    prob: Float or Tensor.
    name: Operation name.

  Returns:
    Tensor of the same shape of `tensor_in`.

  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]`.
  """
  with ops.op_scope([tensor_in], name, "dropout") as name:
    if isinstance(prob, float):
      prob = vs.get_variable("prob", [],
                             initializer=init_ops.constant_initializer(prob),
                             trainable=False)
    ops.add_to_collection(DROPOUTS, prob)
    return contrib_dropout(tensor_in, keep_prob=prob)
