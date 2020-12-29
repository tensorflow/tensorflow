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
"""Contains functions to use mixed precision with the graph rewrite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import config
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale_optimizer as loss_scale_optimizer_v1
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# A mapping between optimizers and the corresponding wrapper class that will be
# used for mixed precision.
_REGISTERED_WRAPPER_OPTIMIZER_CLS = {
    optimizer.Optimizer:
        loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer,
}


def _register_wrapper_optimizer_cls(optimizer_cls, wrapper_optimizer_cls):
  _REGISTERED_WRAPPER_OPTIMIZER_CLS[optimizer_cls] = wrapper_optimizer_cls


def _wrap_optimizer(opt, loss_scale, use_v1_behavior):
  """Wraps an optimizer with a LossScaleOptimizer."""

  for wrapper_optimizer in _REGISTERED_WRAPPER_OPTIMIZER_CLS.values():
    if isinstance(opt, wrapper_optimizer):
      raise ValueError('"opt" must not already be an instance of a {cls}. '
                       '`enable_mixed_precision_graph_rewrite` will '
                       'automatically wrap the optimizer with a '
                       '{cls}.'
                       .format(cls=wrapper_optimizer.__name__))

  for optimizer_cls, wrapper_cls in _REGISTERED_WRAPPER_OPTIMIZER_CLS.items():
    if isinstance(opt, optimizer_cls):
      return wrapper_cls(opt, loss_scale)

  if use_v1_behavior:
    raise ValueError('"opt" must be an instance of a tf.train.Optimizer or a '
                     'tf.keras.optimizers.Optimizer, but got: %s' % opt)
  else:
    raise ValueError('"opt" must be an instance of a '
                     'tf.keras.optimizers.Optimizer, but got: %s' % opt)


@deprecation.deprecated(
    '2020-11-30',
    'Use tf.keras.mixed_precision. There is a guide at '
    'https://www.tensorflow.org/guide/mixed_precision. Alternatively, '
    '`tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite` can '
    'be used, but this is not recommended for TF2 code.')
@tf_export('train.experimental.enable_mixed_precision_graph_rewrite', v1=[])
def enable_mixed_precision_graph_rewrite(opt, loss_scale='dynamic'):
  """Enable mixed precision via a graph rewrite.

  Mixed precision is the use of both float32 and float16 data types when
  training a model to improve performance. This is achieved via a graph rewrite
  operation and a loss-scale optimizer.

  Performing arithmetic operations in float16 takes advantage of specialized
  processing units, such as NVIDIA Tensor Cores, for much higher arithmetic
  throughput. However, due to the smaller representable range, performing the
  entire training with float16 can result in gradient underflow, that is, small
  gradient values becoming zeroes. Instead, performing only select arithmetic
  operations in float16 results in higher throughput and decreased training
  time when using compatible hardware accelerators while also reducing memory
  usage, typically without sacrificing model accuracy.

  Note: While the mixed precision rewrite changes the datatype of various
  layers throughout the model, the same accuracy reached in float32 is
  expected. If a `NaN` gradient occurs with dynamic loss scaling, the model
  update for that batch is skipped. In this case, the global step count is not
  incremented, and the `LossScaleOptimizer` attempts to decrease the loss
  scaling value to avoid `NaN` values in subsequent iterations. This approach
  has been shown to achieve the same accuracy as float32 and, in most cases,
  better training throughput.

  Example:

  ```python
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(64, activation='softmax'),
  ])

  opt = tf.keras.optimizers.SGD()
  opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
  model.compile(loss="mse", optimizer=opt)

  x_train = np.random.random((1024, 64))
  y_train = np.random.random((1024, 64))
  model.fit(x_train, y_train)
  ```

  Calling `enable_mixed_precision_graph_rewrite(opt)` enables the graph rewrite
  operation before computing gradients. The function additionally returns an
  `Optimizer` (`opt`) wrapped with a `LossScaleOptimizer`. This prevents
  underflow in the float16 tensors during the backward pass. An optimizer of
  type `tf.keras.optimizers.Optimizer` or `tf.compat.v1.train.Optimizer` must be
  passed to this function, which will then be wrapped to use loss scaling.

  The graph rewrite operation changes the dtype of certain operations in the
  graph from float32 to float16. There are several categories of operations
  that are either included or excluded by this rewrite operation. The following
  categories of Ops are defined inside corresponding functions under the class
  `AutoMixedPrecisionLists` in
  <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
  core/grappler/optimizers/auto_mixed_precision_lists.h">
  auto_mixed_precision_lists.h</a>:

  * `ClearList`: Ops that do not have numerically significant adverse effects.
  E.g. `ArgMax` and `Floor`.
  * `AllowList`: Ops that are considered numerically safe for execution in
  float16, and thus are always converted. E.g. `Conv2D`.
  * `DenyList`: Ops that are numerically unsafe to execute in float16 and
  can negatively affect downstream nodes. E.g. `Softmax`.
  * `GrayList`: Ops that are considered numerically safe for execution in
  float16 unless downstream from a DenyList Op. E.g. `Add` and `AvgPool`.

  When this function is used, gradients should be computed and applied with the
  returned optimizer, either by calling `opt.minimize()` or
  `opt.compute_gradients()` followed by `opt.apply_gradients()`. If gradients
  are instead computed with `tf.gradients` or `tf.GradientTape`, loss scaling
  will not be applied, which will likely cause your model not to converge due to
  float16 underflow problems. To apply lossing scaling with `tf.gradients` or
  `tf.GradientTape`, `LossScaleOptimizer.get_scaled_loss` and
  `LossScaleOptimizer.get_unscaled_gradients`. See
  `keras.mixed_precision.experimental.LossScaleOptimizer` for details how to do
  this.

  When eager execution is enabled, the mixed precision graph rewrite is only
  enabled within `tf.function`s, as outside `tf.function`s, there is no graph.

  For NVIDIA GPUs with Tensor cores, as a general performance guide, dimensions
  (such as batch size, input size, output size, and channel counts)
  should be powers of two if under 256, or  otherwise divisible by 8 if above
  256. For more information, check out the
  [NVIDIA Deep Learning Performance Guide](
  https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html).

  Currently, mixed precision is only enabled on NVIDIA Tensor Core GPUs with
  Compute Capability 7.0 and above (Volta, Turing, or newer architectures). The
  parts of the graph on CPUs and TPUs are untouched by the graph rewrite.

  ## Comparison with the Keras mixed precision  API
  Both this function and the [Keras mixed precision
  API](https://www.tensorflow.org/guide/keras/mixed_precision) enable the use of
  mixed precision in a model. Therefore, only one of the two APIs can be used.
  We recommend using the Keras mixed precision API, as it is more customizable
  and supports Eager execution. However, it only supports models which use Keras
  layers, while the graph rewrite works in any model that uses `tf.function`s.

  The core difference between the two APIs is that this function is a graph
  rewrite, and so it changes the graph to use mixed precision under the hood.
  You still build your graph in float32, and the graph rewrite will change
  certain ops to float16. The Keras mixed precision API directly builds the
  Keras Model using a mix of float16 and float32.

  One core advantage of the Keras API is it supports mixed precision with Eager
  execution, i.e. mixed precision outside `tf.function`s. The graph rewrite will
  only affect ops within `tf.function`s, making it harder to debug if issues
  occur with mixed precision. The Keras API is also more customizable, as you
  can override any layer to run in float32 by passing `dtype="float32"` to the
  layer constructor. Additionally, you can query the dtype of tensors in the
  model by checking `tensor.dtype`. With the graph rewrite, all tensors appear
  to be float32 since the dtype is only changed under the hood.

  The main advantage of the graph rewrite (this function) is that it works even
  if you do not use Keras layers or any other part of Keras. The Keras mixed
  precision API requires models which use Keras layers, as it only inserts casts
  inside Keras layers and models. Another advantage is that the graph rewrite
  never results in a TypeError, which the Keras API may introduce if you do
  certain operations outside Keras. For example, the following will result in a
  TypeError if the Keras mixed precision API is enabled, as a float16 and
  float32 tensor will be added:
  `tf.keras.layers.Dense(2)(x) + tf.keras.layers.Dense(2, dtype="float32")(x)`

  Raises:
    `ValueError`, if the `tf.keras.mixed_precision` API is also used by calling
    `tf.keras.mixed_precision.experimental.set_policy`. Only one mixed precision
    API can be used.

  Args:
    opt: An instance of a `tf.keras.optimizers.Optimizer`.
    loss_scale: Either an int/float, the string `"dynamic"`, or an instance of a
      `tf.mixed_precision.experimental.LossScale`. The loss scale to use. It is
      recommended to keep this as its default value of `"dynamic"`, which will
      adjust the scaling automatically to prevent `Inf` or `NaN` values.

  Returns:
    A version of `opt` that will use loss scaling to prevent underflow.
  """
  return _enable_mixed_precision_graph_rewrite_base(opt, loss_scale,
                                                    use_v1_behavior=False)


@deprecation.deprecated_endpoints(
    'train.experimental.enable_mixed_precision_graph_rewrite')
@tf_export(v1=['mixed_precision.enable_mixed_precision_graph_rewrite',
               'train.experimental.enable_mixed_precision_graph_rewrite'])
def enable_mixed_precision_graph_rewrite_v1(opt, loss_scale='dynamic'):
  """Enable mixed precision via a graph rewrite.

  Mixed precision is the use of both float32 and float16 data types when
  training a model to improve performance. This is achieved via a graph rewrite
  operation and a loss-scale optimizer.

  Performing arithmetic operations in float16 takes advantage of specialized
  processing units, such as NVIDIA Tensor Cores, for much higher arithmetic
  throughput. However, due to the smaller representable range, performing the
  entire training with float16 can result in gradient underflow, that is, small
  gradient values becoming zeroes. Instead, performing only select arithmetic
  operations in float16 results in higher throughput and decreased training
  time when using compatible hardware accelerators while also reducing memory
  usage, typically without sacrificing model accuracy.

  Note: While the mixed precision rewrite changes the datatype of various
  layers throughout the model, the same accuracy reached in float32 is
  expected. If a `NaN` gradient occurs with dynamic loss scaling, the model
  update for that batch is skipped. In this case, the global step count is not
  incremented, and the `LossScaleOptimizer` attempts to decrease the loss
  scaling value to avoid `NaN` values in subsequent iterations. This approach
  has been shown to achieve the same accuracy as float32 and, in most cases,
  better training throughput.

  Example:

  ```python
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(64, activation='softmax'),
  ])

  opt = tf.keras.optimizers.SGD()
  opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
  model.compile(loss="mse", optimizer=opt)

  x_train = np.random.random((1024, 64))
  y_train = np.random.random((1024, 64))
  model.fit(x_train, y_train)
  ```

  Calling `enable_mixed_precision_graph_rewrite(opt)` enables the graph rewrite
  operation before computing gradients. The function additionally returns an
  `Optimizer` (`opt`) wrapped with a `LossScaleOptimizer`. This prevents
  underflow in the float16 tensors during the backward pass. An optimizer of
  type `tf.train.Optimizer` or `tf.keras.optimizers.Optimizer` must be passed
  to this function, which will then be wrapped to use loss scaling.

  The graph rewrite operation changes the `dtype` of certain operations in the
  graph from float32 to float16. There are several categories of operations
  that are either included or excluded by this rewrite operation. The following
  categories of Ops are defined inside corresponding functions under the class 
  `AutoMixedPrecisionLists` in
  <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
  core/grappler/optimizers/auto_mixed_precision_lists.h">
  auto_mixed_precision_lists.h</a>:

  * `ClearList`: Ops that do not have numerically significant adverse effects.
  E.g. `ArgMax` and `Floor`.
  * `AllowList`: Ops that are considered numerically safe for execution in
  float16, and thus are always converted. E.g. `Conv2D`.
  * `DenyList`: Ops that are numerically unsafe to execute in float16 and
  can negatively affect downstream nodes. E.g. `Softmax`.
  * `GrayList`: Ops that are considered numerically safe for execution in
  float16 unless downstream from a DenyList Op. E.g. `Add` and `AvgPool`.

  When this function is used, gradients should only be computed and applied
  with the returned optimizer, either by calling `opt.minimize()` or
  `opt.compute_gradients()` followed by `opt.apply_gradients()`.
  Gradients should not be computed with `tf.gradients` or `tf.GradientTape`.
  This is because the returned optimizer will apply loss scaling, and
  `tf.gradients` or `tf.GradientTape` will not. If you do directly use
  `tf.gradients` or `tf.GradientTape`, your model may not converge due to
  float16 underflow problems.

  When eager execution is enabled, the mixed precision graph rewrite is only
  enabled within `tf.function`s, as outside `tf.function`s, there is no graph.

  For NVIDIA GPUs with Tensor cores, as a general performance guide, dimensions
  (such as batch size, input size, output size, and channel counts)
  should be powers of two if under 256, or  otherwise divisible by 8 if above
  256. For more information, check out the
  [NVIDIA Deep Learning Performance Guide](
  https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html).

  Currently, mixed precision is only enabled on NVIDIA Tensor Core GPUs with
  Compute Capability 7.0 and above (Volta, Turing, or newer architectures). The
  parts of the graph on CPUs and TPUs are untouched by the graph rewrite.

  Raises:
    `ValueError`, if the `tf.keras.mixed_precision` API is also used by calling
    `tf.keras.mixed_precision.experimental.set_policy`. Only one mixed precision
    API can be used.

  Args:
    opt: An instance of a `tf.keras.optimizers.Optimizer` or a
      `tf.train.Optimizer`.
    loss_scale: Either an int/float, the string `"dynamic"`, or an instance of
      a `tf.mixed_precision.experimental.LossScale`. The loss scale to use. It
      is recommended to keep this as its default value of `"dynamic"`, which
      will adjust the scaling automatically to prevent `Inf` or `NaN` values.

  Returns:
    A version of `opt` that will use loss scaling to prevent underflow.
  """
  # TODO(reedwm): If a ConfigProto is passed to Session, either assert that
  # auto_mixed_precision is on or turn it on for the user.
  return _enable_mixed_precision_graph_rewrite_base(opt, loss_scale,
                                                    use_v1_behavior=True)


def _enable_mixed_precision_graph_rewrite_base(opt, loss_scale,
                                               use_v1_behavior):
  """Enables mixed precision. See `enable_mixed_precision_graph_rewrite`."""
  if mixed_precision_global_state.using_mixed_precision_policy:
    raise ValueError(
        'The mixed precision graph rewrite cannot be enabled, because the '
        'global Keras dtype Policy has been set to a mixed precision policy. '
        'At most, one of the following can be called:\n\n'
        '  1. tf.keras.mixed_precision.experimental.set_policy() with a mixed '
        'precision policy (You called this first)\n\n'
        '  2. tf.train.experimental.enable_mixed_precision_graph_rewrite() '
        '(You called this second)\n'
        'You called both functions, which is an error, because both functions '
        'enable you to use mixed precision. If in doubt which function to use, '
        'use the first, as it supports Eager execution and is more '
        'customizable.')

  if mixed_precision_global_state.non_mixed_precision_session_created:
    # TODO(reedwm): Give the stacktrace of the existing Sessions. And if the
    # Sessions have already been closed, do not raise this error message.
    tf_logging.warn('You already have existing Sessions that do not use mixed '
                    'precision. enable_mixed_precision_graph_rewrite() will '
                    'not affect these Sessions.')
  opt = _wrap_optimizer(opt, loss_scale, use_v1_behavior=use_v1_behavior)
  config.set_optimizer_experimental_options({'auto_mixed_precision': True})
  mixed_precision_global_state.mixed_precision_graph_rewrite_is_enabled = True
  return opt


@deprecation.deprecated(
    '2020-11-30',
    'Use tf.keras.mixed_precision. There is a guide at '
    'https://www.tensorflow.org/guide/mixed_precision. Alternatively, '
    '`tf.compat.v1.mixed_precision.disable_mixed_precision_graph_rewrite` can '
    'be used, but this is not recommended for TF2 code.')
@tf_export('train.experimental.disable_mixed_precision_graph_rewrite', v1=[])
def disable_mixed_precision_graph_rewrite():
  """Disables the mixed precision graph rewrite.

  After this is called, the mixed precision graph rewrite will no longer run for
  tf.functions, and so float32 operations will no longer be converted to
  float16.

  This does not undo the effects of loss scaling. Any optimizers wrapped with a
  LossScaleOptimizer will continue to do loss scaling, although this loss
  scaling will no longer be useful, as the graph rewrite no longer converts
  tf.functions to use float16.

  This function is useful for unit testing. A unit test can test using the mixed
  precision graph rewrite, then disable it so future unit tests continue using
  float32.
  """
  if not mixed_precision_global_state.mixed_precision_graph_rewrite_is_enabled:
    tf_logging.warn('disable_mixed_precision_graph_rewrite() called when mixed '
                    'precision is already disabled.')
  config.set_optimizer_experimental_options({'auto_mixed_precision': False})
  mixed_precision_global_state.mixed_precision_graph_rewrite_is_enabled = False


@deprecation.deprecated_endpoints(
    'train.experimental.disable_mixed_precision_graph_rewrite')
@tf_export(v1=['mixed_precision.disable_mixed_precision_graph_rewrite',
               'train.experimental.disable_mixed_precision_graph_rewrite'])
def disable_mixed_precision_graph_rewrite_v1():
  """Disables the mixed precision graph rewrite.

  After this is called, the mixed precision graph rewrite will no longer run for
  new Sessions, and so float32 operations will no longer be converted to float16
  in such Sessions. However, any existing Sessions will continue to have the
  graph rewrite enabled if they were created after
  `enable_mixed_precision_graph_rewrite` was called but before
  `disable_mixed_precision_graph_rewrite` was called.

  This does not undo the effects of loss scaling. Any optimizers wrapped with a
  LossScaleOptimizer will continue to do loss scaling, although this loss
  scaling will no longer be useful if the optimizer is used in new Sessions, as
  the graph rewrite no longer converts the graph to use float16.

  This function is useful for unit testing. A unit tests can test using the
  mixed precision graph rewrite, then disable it so future unit tests continue
  using float32. If this is done, unit tests should not share a single session,
  as `enable_mixed_precision_graph_rewrite` and
  `disable_mixed_precision_graph_rewrite` have no effect on existing sessions.
  """
  # We only have a separate V1 version of this function, because the V1
  # docstring mentions sessions.
  disable_mixed_precision_graph_rewrite()
