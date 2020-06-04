# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""The V2 implementation of Normalization layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.distribute import reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.experimental.SyncBatchNormalization', v1=[])  # pylint: disable=g-classes-have-attributes
class SyncBatchNormalization(normalization.BatchNormalizationBase):
  r"""Normalize and scale inputs or activations synchronously across replicas.

  Applies batch normalization to activations of the previous layer at each batch
  by synchronizing the global batch statistics across all devices that are
  training the model. For specific details about batch normalization please
  refer to the `tf.keras.layers.BatchNormalization` layer docs.

  If this layer is used when using tf.distribute strategy to train models
  across devices/workers, there will be an allreduce call to aggregate batch
  statistics across all replicas at every training step. Without tf.distribute
  strategy, this layer behaves as a regular `tf.keras.layers.BatchNormalization`
  layer.

  Example usage:
  ```
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.experimental.SyncBatchNormalization())
  ```

  Arguments:
    axis: Integer, the axis that should be normalized
      (typically the features axis).
      For instance, after a `Conv2D` layer with
      `data_format="channels_first"`,
      set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
      If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
      When the next layer is linear (also e.g. `nn.relu`),
      this can be disabled since the scaling
      will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use [Batch Renormalization](
      https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    trainable: Boolean, if `True` the variables will be marked as trainable.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode.
      - `training=True`: The layer will normalize its inputs using the
        mean and variance of the current batch of inputs.
      - `training=False`: The layer will normalize its inputs using the
        mean and variance of its moving statistics, learned during training.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.

  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               trainable=True,
               adjustment=None,
               name=None,
               **kwargs):

    # Currently we only support aggregating over the global batch size.
    super(SyncBatchNormalization, self).__init__(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=False,
        trainable=trainable,
        virtual_batch_size=None,
        name=name,
        **kwargs)

  def _calculate_mean_and_var(self, x, axes, keep_dims):

    with ops.name_scope('moments', values=[x, axes]):
      # The dynamic range of fp16 is too limited to support the collection of
      # sufficient statistics. As a workaround we simply perform the operations
      # on 32-bit floats before converting the mean and variance back to fp16
      y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
      replica_ctx = ds.get_replica_context()
      if replica_ctx:
        local_sum = math_ops.reduce_sum(y, axis=axes, keepdims=True)
        local_squared_sum = math_ops.reduce_sum(math_ops.square(y), axis=axes,
                                                keepdims=True)
        batch_size = math_ops.cast(array_ops.shape_v2(y)[0], dtypes.float32)
        y_sum, y_squared_sum, global_batch_size = (
            replica_ctx.all_reduce(reduce_util.ReduceOp.SUM, [
                local_sum, local_squared_sum, batch_size]))

        axes_vals = [(array_ops.shape_v2(y))[i] for i in range(1, len(axes))]
        multiplier = math_ops.cast(math_ops.reduce_prod(axes_vals),
                                   dtypes.float32)
        multiplier = multiplier * global_batch_size

        mean = y_sum / multiplier
        y_squared_mean = y_squared_sum / multiplier
        # var = E(x^2) - E(x)^2
        variance = y_squared_mean - math_ops.square(mean)
      else:
        # Compute true mean while keeping the dims for proper broadcasting.
        mean = math_ops.reduce_mean(y, axes, keepdims=True, name='mean')
        # sample variance, not unbiased variance
        # Note: stop_gradient does not change the gradient that gets
        #       backpropagated to the mean from the variance calculation,
        #       because that gradient is zero
        variance = math_ops.reduce_mean(
            math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
            axes,
            keepdims=True,
            name='variance')
      if not keep_dims:
        mean = array_ops.squeeze(mean, axes)
        variance = array_ops.squeeze(variance, axes)
      if x.dtype == dtypes.float16:
        return (math_ops.cast(mean, dtypes.float16),
                math_ops.cast(variance, dtypes.float16))
      else:
        return (mean, variance)


@keras_export('keras.layers.BatchNormalization', v1=[])  # pylint: disable=missing-docstring
class BatchNormalization(normalization.BatchNormalizationBase):

  __doc__ = normalization.replace_in_base_docstring([
      ('{{TRAINABLE_ATTRIBUTE_NOTE}}',
       '''
  **About setting `layer.trainable = False` on a `BatchNormalization` layer:**

  The meaning of setting `layer.trainable = False` is to freeze the layer,
  i.e. its internal state will not change during training:
  its trainable weights will not be updated
  during `fit()` or `train_on_batch()`, and its state updates will not be run.

  Usually, this does not necessarily mean that the layer is run in inference
  mode (which is normally controlled by the `training` argument that can
  be passed when calling a layer). "Frozen state" and "inference mode"
  are two separate concepts.

  However, in the case of the `BatchNormalization` layer, **setting
  `trainable = False` on the layer means that the layer will be
  subsequently run in inference mode** (meaning that it will use
  the moving mean and the moving variance to normalize the current batch,
  rather than using the mean and variance of the current batch).

  This behavior has been introduced in TensorFlow 2.0, in order
  to enable `layer.trainable = False` to produce the most commonly
  expected behavior in the convnet fine-tuning use case.

  Note that:
    - This behavior only occurs as of TensorFlow 2.0. In 1.*,
      setting `layer.trainable = False` would freeze the layer but would
      not switch it to inference mode.
    - Setting `trainable` on an model containing other layers will
      recursively set the `trainable` value of all inner layers.
    - If the value of the `trainable`
      attribute is changed after calling `compile()` on a model,
      the new value doesn't take effect for this model
      until `compile()` is called again.
      ''')])

  _USE_V2_BEHAVIOR = True
