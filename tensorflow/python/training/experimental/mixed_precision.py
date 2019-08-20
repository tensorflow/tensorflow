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
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


def _wrap_optimizer(opt, loss_scale, use_v1_behavior):
  """Wraps an optimizer with a LossScaleOptimizer."""

  if isinstance(opt, loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer):
    raise ValueError('"opt" must not already be an instance of a '
                     'MixedPrecisionLossScaleOptimizer. '
                     '`enable_mixed_precision_graph_rewrite` will '
                     'automatically wrap the optimizer with a '
                     'MixedPrecisionLossScaleOptimizer.')
  # To avoid a circular dependency, we cannot depend on tf.keras. Because
  # LossScaleOptimizer is in Keras, we cannot use isinstance, so instead check
  # the class name.
  if opt.__class__.__name__ == 'LossScaleOptimizer':
    raise ValueError('"opt" must not already be an instance of a '
                     'LossScaleOptimizer. '
                     '`enable_mixed_precision_graph_rewrite` will '
                     'automatically wrap the optimizer with a '
                     'LossScaleOptimizer.')

  if isinstance(opt, optimizer.Optimizer):
    # For convenience, we allow the V2 version of this function to wrap the V1
    # optimizer, even though we do not document this.
    return loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer(opt,
                                                                    loss_scale)

  # Because we cannot depend on tf.keras, we see if `opt` is an instance of the
  # Keras OptimizerV2 class by checking the subclass names.
  base_classes = tf_inspect.getmro(opt.__class__)
  base_class_names = [cls.__name__ for cls in base_classes]
  is_loss_scale_optimizer_v2 = 'OptimizerV2' in base_class_names

  if is_loss_scale_optimizer_v2:
    # Because we cannot depend on tf.keras, we cannot unconditionally do this
    # import. But since `opt` is a Keras OptimizerV2, we know keras is
    # importable, so it is safe to do this import. (Technically, it's possible
    # to have a dependency on OptimizerV2 and not LossScaleOptimizer, but this
    # is not done in practice).
    from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as loss_scale_optimizer_v2  # pylint: disable=g-import-not-at-top
    return loss_scale_optimizer_v2.LossScaleOptimizer(opt, loss_scale)

  if use_v1_behavior:
    raise ValueError('"opt" must be an instance of a tf.train.Optimizer or a '
                     'tf.keras.optimizers.Optimizer, but got: %s' % opt)
  else:
    raise ValueError('"opt" must be an instance of a '
                     'tf.keras.optimizers.Optimizer, but got: %s' % opt)


@tf_export('train.experimental.enable_mixed_precision_graph_rewrite', v1=[])
def enable_mixed_precision_graph_rewrite(opt, loss_scale='dynamic'):
  """Enable mixed precision via a graph rewrite.

  Mixed precision is the use of both float32 and float16 data types when
  training a model to improve performance. This is achieved via a graph rewrite
  operation and a loss-scale optimizer.

  Performing arithmetic operations in float16 takes advantage of specialized
  processing units, such as NVIDIA Tensor Cores for much higher arithmetic
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
    ...
  ])

  opt = tf.keras.optimizers.SGD()
  opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

  model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

  model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)
  ```

  For a complete example showing the speed-up on training an image
  classification task on CIFAR10, check out this
  <a href="https://colab.research.google.com/github/NVIDIA/
  DeepLearningExamples/blob/master/TensorFlow/docs/amp/notebook_v1.14/
  auto_mixed_precision_demo_cifar10.ipynb">Colab notebook</a>.

  Calling `enable_mixed_precision_graph_rewrite(opt)` enables the graph rewrite
  operation before computing gradients. The function additionally returns an
  `Optimizer`(`opt`) wrapped with a `LossScaleOptimizer`. This prevents
  underflow in the float16 tensors during the backward pass. An optimizer of
  type `tf.train.Optimizer` or `tf.keras.optimizers.Optimizer` must be passed
  to this function, which will then be wrapped to use loss scaling.

  <img src="
  http://developer.download.nvidia.com/compute/machine-learning/frameworks/
  TF_mixed_precision_training.png" width="500px">

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
  * `WhiteList`: Ops that are considered numerically safe for execution in
  float16, and thus are always converted. E.g. `Conv2D`.
  * `BlackList`: Ops that are numerically unsafe to execute in float16 and
  can negatively affect downstream nodes. E.g. `Softmax`.
  * `GrayList`: Ops that are considered numerically safe for execution in
  float16 unless downstream from a BlackList Op. E.g. `Add` and `AvgPool`.

  When this function is used, gradients should only be computed and applied
  with the returned optimizer, either by calling `opt.minimize()` or
  `opt.compute_gradients()` followed by `opt.apply_gradients()`.
  Gradients should not be computed with `tf.gradients` or `tf.GradientTape`.
  This is because the returned optimizer will apply loss scaling, and
  `tf.gradients` or `tf.GradientTape` will not. If you do directly use
  `tf.gradients` or `tf.GradientTape`, your model may not converge due to
  float16 underflow problems.

  When eager execution is enabled, the mixed precision graph rewrite is only
  enabled within `tf.function`, as outside `tf.function`, there is no graph.

  For NVIDIA GPUs with Tensor cores, as a general performance guide, dimensions
  (such as batch size, input size, output size, and channel counts)
  should be powers of two if under 256, or  otherwise divisible by 8 if above
  256. For more information, check out the
  [NVIDIA Deep Learning Performance Guide](
  https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html).

  Currently, mixed precision is only enabled on NVIDIA Tensor Core GPUs with
  Compute Capability 7.0 and above (Volta, Turing, or newer architectures). The
  parts of the graph on CPUs and TPUs are untouched by the graph rewrite. TPU
  support is coming soon. CPUs are not supported, as CPUs do not run float16
  operations faster than float32 operations.

  Raises:
    `ValueError` when
    `mixed_precision_global_state.using_default_mixed_precision_policy`
    is set to `False` before
    `tf.train.experimental.enable_mixed_precision_graph_rewrite()`
    is called.

  Args:
    opt: An instance of a `tf.keras.optimizers.Optimizer`.
    loss_scale: Either an int/float, the string `"dynamic"`, or an instance of a
      `tf.train.experimental.LossScale`. The loss scale to use. It is
      recommended to keep this as its default value of `"dynamic"`, which will
      adjust the scaling automatically to prevent `Inf` or `NaN` values.

  Returns:
    A version of `opt` that will use loss scaling to prevent underflow.
  """
  return _enable_mixed_precision_graph_rewrite_base(opt, loss_scale,
                                                    use_v1_behavior=False)


@tf_export(v1=['train.experimental.enable_mixed_precision_graph_rewrite'])
def enable_mixed_precision_graph_rewrite_v1(opt, loss_scale='dynamic'):
  """Enable mixed precision via a graph rewrite.

  Mixed precision is the use of both float32 and float16 data types when
  training a model to improve performance. This is achieved via a graph rewrite
  operation and a loss-scale optimizer.
  
  Performing arithmetic operations in float16 takes advantage of specialized
  processing units, such as NVIDIA Tensor Cores for much higher arithmetic
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
    ...
  ])
  
  opt = tf.keras.optimizers.SGD()
  opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
  
  model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])
  
  model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)
  ```
  
  For a complete example showing the speed-up on training an image
  classification task on CIFAR10, check out this
  <a href="https://colab.research.google.com/github/NVIDIA/
  DeepLearningExamples/blob/master/TensorFlow/docs/amp/notebook_v1.14/
  auto_mixed_precision_demo_cifar10.ipynb">Colab notebook</a>.
  
  Calling `enable_mixed_precision_graph_rewrite(opt)` enables the graph rewrite
  operation before computing gradients. The function additionally returns an
  `Optimizer`(`opt`) wrapped with a `LossScaleOptimizer`. This prevents
  underflow in the float16 tensors during the backward pass. An optimizer of
  type `tf.train.Optimizer` or `tf.keras.optimizers.Optimizer` must be passed
  to this function, which will then be wrapped to use loss scaling.
  
  <img src="
  http://developer.download.nvidia.com/compute/machine-learning/frameworks/
  TF_mixed_precision_training.png" width="500px">
  
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
  * `WhiteList`: Ops that are considered numerically safe for execution in
  float16, and thus are always converted. E.g. `Conv2D`.
  * `BlackList`: Ops that are numerically unsafe to execute in float16 and
  can negatively affect downstream nodes. E.g. `Softmax`.
  * `GrayList`: Ops that are considered numerically safe for execution in
  float16 unless downstream from a BlackList Op. E.g. `Add` and `AvgPool`.
  
  When this function is used, gradients should only be computed and applied
  with the returned optimizer, either by calling `opt.minimize()` or
  `opt.compute_gradients()` followed by `opt.apply_gradients()`.
  Gradients should not be computed with `tf.gradients` or `tf.GradientTape`.
  This is because the returned optimizer will apply loss scaling, and
  `tf.gradients` or `tf.GradientTape` will not. If you do directly use
  `tf.gradients` or `tf.GradientTape`, your model may not converge due to
  float16 underflow problems.
  
  When eager execution is enabled, the mixed precision graph rewrite is only
  enabled within `tf.function`, as outside `tf.function`, there is no graph.
  
  For NVIDIA GPUs with Tensor cores, as a general performance guide, dimensions
  (such as batch size, input size, output size, and channel counts)
  should be powers of two if under 256, or  otherwise divisible by 8 if above
  256. For more information, check out the
  [NVIDIA Deep Learning Performance Guide](
  https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html).
  
  Currently, mixed precision is only enabled on NVIDIA Tensor Core GPUs with
  Compute Capability 7.0 and above (Volta, Turing, or newer architectures). The
  parts of the graph on CPUs and TPUs are untouched by the graph rewrite. TPU
  support is coming soon. CPUs are not supported, as CPUs do not run float16
  operations faster than float32 operations.
  
  Raises:
    `ValueError` when
    `mixed_precision_global_state.using_default_mixed_precision_policy`
    is set to `False` before
    `tf.train.experimental.enable_mixed_precision_graph_rewrite()`
    is called.
  
  Args:
    opt: An instance of a `tf.keras.optimizers.Optimizer` or a
      `tf.train.Optimizer`.
    loss_scale: Either an int/float, the string `"dynamic"`, or an instance of
      a `tf.train.experimental.LossScale`. The loss scale to use. It is
      recommended to keep this as its default value of `"dynamic"`, which will 
      adjust the scaling automatically to prevent `Inf` or `NaN` values.
  
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
  if not mixed_precision_global_state.using_default_mixed_precision_policy:
    raise ValueError(
        'The mixed precision graph rewrite cannot be enabled, because a keras '
        'mixed precision Policy has been set. At most, one of the following '
        'functions can be called:\n\n'
        '  1. tf.keras.mixed_precision.experimental.set_policy() (You called '
        'this first)\n'
        '  2. tf.train.experimental.enable_mixed_precision_graph_rewrite() '
        '(You called this second)\n\n'
        'You called both functions, which is an error, because both functions '
        'enable you to use mixed precision. If in doubt which function to use, '
        'use the second, as it is currently more complete and easy to use. The '
        'second function enables mixed precision in the graph with a graph '
        'rewrite. However it is currently not very customizable, and does not '
        'support eager.')

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


@tf_export(v1=['train.experimental.disable_mixed_precision_graph_rewrite'])
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
