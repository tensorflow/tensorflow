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
from tensorflow.python.training import optimizer
from tensorflow.python.training.experimental import loss_scale_optimizer as loss_scale_optimizer_v1
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
  """Enable mixed precision in `tf.function`s via a graph rewrite.

  Mixed precision is the use of both float16 and float32 when training a model,
  and is used to make the model run faster. This function will use mixed
  precision to speed up the execution time of `tf.function`s when run on a GPU.
  It does this by changing the dtype of certain operations in the function's
  graph from float32 to float16.

  This function additionally wraps an Optimizer with a LossScaleOptimizer, which
  is required to prevent underflow in the float16 tensors during the backwards
  pass. An optimizer must be passed to this function, which will then be wrapped
  to use loss scaling.

  When this function is used, gradients should only be computed and applied with
  the returned optimizer through `opt.minimize()`, and not with a
  `tf.GradientTape`. This is because the returned optimizer will apply loss
  scaling, and `tf.GradientTape` will not. If you do use a `tf.GradientTape`,
  your model may train to a worse quality.

  Currently, mixed precision is only enabled on Volta GPUs and above. TPU
  support is coming soon. CPUs are not supported, as CPUs do not run float16
  operations faster than float32 operations.

  Args:
    opt: An instance of a `tf.keras.optimizers.Optimizer`.
    loss_scale: Either an int/float, the string "dynamic", or an instance of a
      `tf.keras.mixed_precision.experimental.LossScale`. The loss scale to use.
      It is recommended to keep this as it's default value of "dynamic".

  Returns:
    A version of `opt` that will use loss scaling to prevent underflow.
  """
  return _enable_mixed_precision_graph_rewrite_base(opt, loss_scale,
                                                    use_v1_behavior=False)


@tf_export(v1=['train.experimental.enable_mixed_precision_graph_rewrite'])
def enable_mixed_precision_graph_rewrite_v1(opt, loss_scale='dynamic'):
  """Enable mixed precision via a graph rewrite.

  Mixed precision is the use of both float16 and float32 when training a model,
  and is used to make the model run faster. This function will use mixed
  precision to speed up the execution time of your model when run on a GPU. It
  does this by changing the dtype of certain operations in the graph from
  float32 to float16.

  This function additionally wraps an Optimizer with a LossScaleOptimizer, which
  is required to prevent underflow in the float16 tensors during the backwards
  pass. An optimizer must be passed to this function, which will then be wrapped
  to use loss scaling.

  When this function is used, gradients should only be computed and applied with
  the returned optimizer, either by calling `opt.minimize()` or
  `opt.compute_gradients()` followed by `opt.apply_gradients()`. Gradients
  should not be computed with `tf.gradients` or `tf.GradientTape`. This is
  because the returned optimizer will apply loss scaling, and
  `tf.gradients`/`tf.GradientTape` will not. If you do directly use
  `tf.gradients` or `tf.GradientTape`, your model may train to a worse quality.

  Note: If you explicitly pass a ConfigProto to your Session, you must set the
  `auto_mixed_precision` option to ON. If you do not pass any ConfigProto to
  your Session, no extra work needs to be done. For example:

  ```
  loss, trainable_vars = ...
  opt = tf.keras.optimizers.SGD(0.001)
  opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
  train_op = opt.minimize(loss, vars=trainable_vars)

  # No extra work needs to be done, as no ConfigProto is passed to the Session
  with tf.Session() as sess:
    sess.run(train_op)

  # If a ConfigProto is passed to Session, you MUST set the
  # `auto_mixed_precision` field to ON.
  config = tf.ConfigProto()
  from tensorflow.core.protobuf import rewriter_config_pb2
  config.graph_options.rewrite_options.auto_mixed_precision = (
      rewriter_config_pb2.RewriterConfig.ON)
  with tf.Session(config=config) as sess:
    sess.run(train_op)
  ```

  Currently, mixed precision is only enabled on Volta GPUs and above. TPU
  support is coming soon. CPUs are not supported, as CPUs do not run float16
  operations faster than float32 operations.

  Args:
    opt: An instance of a `tf.keras.optimizers.Optimizer` or a
      `tf.train.Optimizer`.
    loss_scale: Either an int/float, the string "dynamic", or an instance of a
      `tf.keras.mixed_precision.experimental.LossScale`. The loss scale to use.
      It is recommended to keep this as it's default value of "dynamic".

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
  opt = _wrap_optimizer(opt, loss_scale, use_v1_behavior=use_v1_behavior)
  config.set_optimizer_experimental_options({'auto_mixed_precision': True})
  return opt
