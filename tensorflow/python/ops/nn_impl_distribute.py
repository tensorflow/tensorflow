# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Implementation of Neural Net (NN) functions with distribution strategy."""

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export("nn.scale_regularization_loss")
@dispatch.add_dispatch_support
def scale_regularization_loss(regularization_loss):
  """Scales the sum of the given regularization losses by number of replicas.

  Usage with distribution strategy and custom training loop:

  ```python
  with strategy.scope():
    def compute_loss(self, label, predictions):
      per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)

      # Compute loss that is scaled by sample_weight and by global batch size.
      loss = tf.nn.compute_average_loss(
          per_example_loss,
          sample_weight=sample_weight,
          global_batch_size=GLOBAL_BATCH_SIZE)

      # Add scaled regularization losses.
      loss += tf.nn.scale_regularization_loss(tf.nn.l2_loss(weights))
      return loss
  ```

  Args:
    regularization_loss: Regularization loss.

  Returns:
    Scalar loss value.
  """  # pylint: disable=g-doc-exception
  if (
      distribute_lib.has_strategy()
      and distribute_lib.in_cross_replica_context()
  ):
    raise RuntimeError(
        "You are calling `scale_regularization_loss` in cross replica context, "
        "while it was expected to be called in replica context."
    )

  num_replicas = distribute_lib.get_strategy().num_replicas_in_sync
  return math_ops.reduce_sum(regularization_loss) / num_replicas


@tf_export("nn.compute_average_loss")
@dispatch.add_dispatch_support
def compute_average_loss(
    per_example_loss, sample_weight=None, global_batch_size=None
):
  """Scales per-example losses with sample_weights and computes their average.

  Usage with distribution strategy and custom training loop:

  ```python
  with strategy.scope():
    def compute_loss(labels, predictions, sample_weight=None):

      # If you are using a `Loss` class instead, set reduction to `NONE` so that
      # we can do the reduction afterwards and divide by global batch size.
      per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)

      # Compute loss that is scaled by sample_weight and by global batch size.
      return tf.nn.compute_average_loss(
          per_example_loss,
          sample_weight=sample_weight,
          global_batch_size=GLOBAL_BATCH_SIZE)
  ```

  Args:
    per_example_loss: Per-example loss.
    sample_weight: Optional weighting for each example.
    global_batch_size: Optional global batch size value. Defaults to (size of
      first dimension of `losses`) * (number of replicas).

  Returns:
    Scalar loss value, obtained by summing the `per_example_loss` and dividing
    by `global_batch_size`. If `global_batch_size` is zero, the result is zero.
  """  # pylint: disable=g-doc-exception
  per_example_loss = ops.convert_to_tensor(per_example_loss)
  input_dtype = per_example_loss.dtype

  with losses_util.check_per_example_loss_rank(per_example_loss):
    if sample_weight is not None:
      sample_weight = ops.convert_to_tensor(sample_weight)
      per_example_loss = losses_util.scale_losses_by_sample_weight(
          per_example_loss, sample_weight
      )
    per_example_loss = math_ops.cast(per_example_loss, input_dtype)

    if global_batch_size is None:
      if (
          distribute_lib.has_strategy()
          and distribute_lib.in_cross_replica_context()
      ):
        raise RuntimeError(
            "You are calling `compute_average_loss` in cross replica context, "
            "while it was expected to be called in replica context."
        )

      num_replicas = distribute_lib.get_strategy().num_replicas_in_sync
      per_replica_batch_size = array_ops.shape_v2(per_example_loss)[0]
      global_batch_size = per_replica_batch_size * num_replicas

    check_ops.assert_scalar_v2(
        global_batch_size, message="global_batch_size must be scalar."
    )
    check_ops.assert_integer_v2(
        global_batch_size, message="global_batch_size must be an integer."
    )
    check_ops.assert_non_negative_v2(
        global_batch_size, message="global_batch_size must be non-negative."
    )

    loss = math_ops.reduce_sum(per_example_loss)
    global_batch_size = math_ops.cast(global_batch_size, input_dtype)
    return math_ops.div_no_nan(loss, global_batch_size)
