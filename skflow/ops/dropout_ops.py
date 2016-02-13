"""Dropout operations and handling."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import division, print_function, absolute_import

import tensorflow as tf


# Key to collect dropout probabilities.
DROPOUTS = "dropouts"


def dropout(tensor_in, prob, name=None):
    """Adds dropout node and stores probability tensor into graph collection.

    Args:
        tensor_in: Input tensor.
        prob: Float or Tensor.

    Returns:
        Tensor of the same shape of `tensor_in`.

    Raises:
        ValueError: If `keep_prob` is not in `(0, 1]`.
    """
    with tf.op_scope([tensor_in], name, "dropout") as name:
        if isinstance(prob, float):
            prob = tf.get_variable("prob", [],
                                   initializer=tf.constant_initializer(prob),
                                   trainable=False)
        tf.add_to_collection(DROPOUTS, prob)
        return tf.nn.dropout(tensor_in, prob)

