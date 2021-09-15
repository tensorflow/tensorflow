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


@tf_export('__internal__.mixed_precision.register_loss_scale_wrapper', v1=[])
def register_loss_scale_wrapper(optimizer_cls, wrapper_cls):
  """Registers a loss scale optimizer wrapper.

  `tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite`
  automatically wraps an optimizer with an optimizer wrapper that performs loss
  scaling. This function registers a `(base_optimizer, wrapper_optimizer)` pair
  that is used by `enable_mixed_precision_graph_rewrite`, where
  `wrapper_optimizer` wraps a `base_optimizer` and applies loss scaling.

  Args:
    optimizer_cls: A base optimizer class, e.g. `tf.keras.optimizers.Optimizer`.
    wrapper_cls: A wrapper that wraps `optimizer_cls` and applies loss scaling,
      e.g. `tf.compat.v1.keras.mixed_precision.LossScaleOptimizer`. The
      constructor should take two arguments: The inner optimizer and a
      `tf.compat.v1.mixed_precision.LossScale`.
  """
  _REGISTERED_WRAPPER_OPTIMIZER_CLS[optimizer_cls] = wrapper_cls


def _wrap_optimizer(opt, loss_scale):
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

  raise ValueError('"opt" must be an instance of a tf.train.Optimizer or a '
                   'tf.keras.optimizers.Optimizer, but got: %s' % opt)


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
    `tf.keras.mixed_precision.set_global_policy`. Only one mixed precision
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
  if mixed_precision_global_state.is_using_mixed_precision_policy():
    raise ValueError(
        'The mixed precision graph rewrite cannot be enabled, because the '
        'global Keras dtype Policy has been set to a mixed precision policy. '
        'At most, one of the following can be called:\n\n'
        '  1. tf.keras.mixed_precision.set_global_policy() with a mixed '
        'precision policy (You called this first)\n\n'
        '  2. tf.train.experimental.enable_mixed_precision_graph_rewrite() '
        '(You called this second)\n'
        'You called both functions, which is an error, because both functions '
        'enable you to use mixed precision. If in doubt which function to use, '
        'use the first, as it supports Eager execution and is more '
        'customizable.')

  if mixed_precision_global_state.non_mixed_precision_session_created():
    # TODO(reedwm): Give the stacktrace of the existing Sessions. And if the
    # Sessions have already been closed, do not raise this error message.
    tf_logging.warn('You already have existing Sessions that do not use mixed '
                    'precision. enable_mixed_precision_graph_rewrite() will '
                    'not affect these Sessions.')
  opt = _wrap_optimizer(opt, loss_scale)
  config.set_optimizer_experimental_options({'auto_mixed_precision': True})
  mixed_precision_global_state.set_mixed_precision_graph_rewrite_enabled(True)
  return opt


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
  if (not
      mixed_precision_global_state.is_mixed_precision_graph_rewrite_enabled()):
    tf_logging.warn('disable_mixed_precision_graph_rewrite() called when mixed '
                    'precision is already disabled.')
  config.set_optimizer_experimental_options({'auto_mixed_precision': False})
  mixed_precision_global_state.set_mixed_precision_graph_rewrite_enabled(False)
