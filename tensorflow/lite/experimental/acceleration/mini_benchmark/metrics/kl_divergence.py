# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""KL-divergence metrics calculation."""

# TODO(b/152872335): (re-)port to tf v2 after output names are kept during
# conversion in v2.
import tensorflow.compat.v1 as tf


def symmetric_kl_divergence(predicted, actual):
  """Calculate symmetric KL-divergence over two classification tensors.

  Note that here the classifications do not form a probability distribution.
  They are, however normalized to 0..1 and calculating a KL-divergence over them
  gives reasonable numerical results.

  Shape of the two inputs must be the same at inference time but is otherwise
  unconstrained.

  Args:
    predicted: classification outputs from model
    actual: golden classification outputs

  Returns:
    Single scalar tensor with symmetric KL-divergence between predicted and
    actual.
  """
  epsilon = tf.constant(1e-7, dtype=tf.float32, name='epsilon')
  p = tf.math.maximum(predicted, epsilon)
  q = tf.math.maximum(actual, epsilon)
  kld_1 = tf.math.reduce_sum(
      tf.math.multiply(p, tf.math.log(tf.math.divide(p, q))))
  kld_2 = tf.math.reduce_sum(
      tf.math.multiply(q, tf.math.log(tf.math.divide(q, p))))
  return tf.add(kld_1, kld_2)
