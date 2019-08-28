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
"""Contains the Policy class for mixed precision training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util.tf_export import keras_export


# Default value of certain arguments, indicating the default behavior for
# that argument should be used.
USE_DEFAULT = 'USE_DEFAULT'


@keras_export('keras.mixed_precision.experimental.Policy')
class Policy(object):
  """A dtype policy for a Keras layer.

  A dtype policy determines dtype-related aspects of a layer, such as its
  computation and variable dtypes. Each layer has a policy. Policies can be
  passed to the 'dtype' argument of layer constructors, or a global policy can
  be set with 'tf.keras.mixed_precision.experimental.set_policy'. A layer will
  default to the global policy if no policy is passed to it's constructor.

  For most models, each layer will have the same computation dtype and variable
  dtype, which will typically be float32. However, when mixed precision
  training is used, most layers will instead have a float16 computation dtype
  and a float32 variable dtype. See [this
  link](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
  for more information on mixed precision training. When the variable dtype does
  not match the computation dtype, variables will be automatically casted to the
  computation dtype to avoid type errors.

  Policies also have a `tf.train.experimental.LossScale` instance, which is used
  by Models to performance loss scaling. Layers which are not Models ignore
  the loss scale.

  Policies are constructed by passing a string to the constructor, e.g.
  `tf.keras.mixed_precision.experimental.Policy('float32')`. The string
  determines the compute and variable dtypes. Currently, it can be one of
  in one of the following forms:

    * Any dtype name, such as 'float32' or 'float64'. Both the variable and
      compute dtypes will be that dtype.
    * '<dtype>_with_float32_vars', where <dtype> is any dtype. The compute dtype
      will be <dtype>, while the variable dtype is float32. This can be used for
      mixed precision, which uses float16 or bfloat16 for most computations, and
      float32 for variables, but it is recommended to use the 'mixed_float16' or
      'mixed_bfloat16' policies instead.
    * 'mixed_float16' or 'mixed_bfloat16': Similar to
      'float16_with_float32_vars' or 'bfloat16_with_float32_vars' respectively.
      'mixed_float16' is identical to 'float16_with_float32_vars' except the
      loss_scale is dynamic by default. 'mixed_bfloat16' is currently identical
      to 'bfloat16_with_float32_vars'. More changes may be added to these mixed
      policies in the future, to further differentiate them from
      [b]float16_with_float32_vars.

  ### How to use mixed precision in layers with Policies

  To use mixed precision in a model, the 'mixed_float16' policy can
  be used. `tf.keras.mixed_precision.experimental.set_policy` can be used to set
  the default policy for layers if no policy is passed to them. For example:

  ```python
  tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
  model = tf.keras.models.Sequential(
      tf.keras.layers.Input((100,)),
      # Dense layers use global policy of 'mixed_float16', which does
      # computations in float16 while keeping variables in float32.
      tf.keras.layers.Dense(10),
      tf.keras.layers.Dense(10),
      # Softmax should be done in float32 for numeric stability. We pass
      # dtype='float32' to use float32 instead of the global policy.
      tf.keras.layers.Activation('Softmax', dtype='float32')
  )
  model.fit(...)  # Train `model`
  ```

  Alternatively, the policy can be passed to individual layers instead of
  setting the global policy with `set_policy`:

  ```python
  policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
  model = tf.keras.models.Sequential(
      tf.keras.layers.Input((100,)),
      tf.keras.layers.Dense(10, dtype=policy),
      tf.keras.layers.Dense(10, dtype=policy),
      # Softmax should be done in float32 for numeric stability.
      tf.keras.layers.Activation('Softmax', dtype='float32')
  )
  model.fit(...)  # Train `model`
  ```

  As the above example shows, strings can be directly passed to layer
  constructors in the `dtype` argument instead of policies, but only if the
  string is convertible to a dtype.

  ### The deprecated "infer" policy

  In addition to a dtype or "<dtype>_with_float32_vars", a policy can also be
  "infer". This Policy is deprecated, and it is not recommended. When a layer
  has an infer policy, it will infer the computation and variable dtype from
  the first input the first time the layer is called.

  Once the layer is called for the first time, the layer's policy will change to
  the dtype of the first input.

  Similarly to "infer", there is a deprecated "infer_with_float32_vars" policy
  that infers the compute dtype, but not the variable dtype.

  In TensorFlow 1, only the "infer" and "infer_with_float32_vars" policies are
  available.
  """
  # TODO(reedwm): Replace link in above docstring with a version that is more
  # TensorFlow-specific, and that also mentions bfloat16.

  def __init__(self, name, loss_scale=USE_DEFAULT):
    """Constructs the policy.

    The `name` argument determines the compute and variable dtype, and has no
    additional effect on the Policy. The compute and variable dtypes can only be
    specified through `name`, and cannot be specified directly.

    Args:
      name: A string. Can be one of the following values:
        * Any dtype name, such as 'float32' or 'float64'. Both the variable and
          compute dtypes will be that dtype.
        * '<dtype>_with_float32_vars', where <dtype> is any dtype. The compute
          dtype will be <dtype>, while the variable dtype is float32. This can
          be used for mixed precision, which uses float16 or bfloat16 for most
          computations, and float32 for variables, but it is recommended to use
          the 'mixed_float16' or 'mixed_bfloat16' policies instead.
        * 'mixed_float16' or 'mixed_bfloat16': Similar to
          'float16_with_float32_vars' or 'bfloat16_with_float32_vars'
          respectively. 'mixed_float16' is identical to
          'float16_with_float32_vars' except the loss_scale is dynamic by
          default. 'mixed_bfloat16' is currently identical to
          'bfloat16_with_float32_vars'. More changes may be added to these mixed
          policies in the future, to further differentiate them from
          [b]float16_with_float32_vars.
        * 'infer' or 'infer_with_float32_vars' (deprecated): Infer the
          computation dtype from the input dtype.
      loss_scale: A `tf.train.experimental.LossScale`, or a value convertible to
        one such as "dynamic". Defaults to using no loss scaling unless `name`
        is "mixed_float16", in which case this defaults to "dynamic". Only
        `tf.keras.Model`s, not layers, use the loss scale, and it is only used
        during `Model.fit` or `Model.train_on_batch`.

    """
    if isinstance(name, dtypes.DType):
      raise TypeError("'name' must be a string, not a DType. "
                      "Instead, pass DType.name. Got: %s" % (name.name,))
    elif not isinstance(name, six.string_types):
      raise TypeError("'name' must be a string, but got: %s" % (name,))
    if name == 'infer_float32_vars':
      # For backwards compatibility. TODO(reedwm): Remove this.
      name = 'infer_with_float32_vars'
    if name == 'float32_with_float32_vars':
      # Doesn't affect correctness, but causes "float32" instead of
      # "float32_with_float32_vars" to be printed in __repr__.
      name = 'float32'
    self._name = name
    self._compute_dtype, self._variable_dtype = self._parse_name(name)

    if loss_scale == USE_DEFAULT:
      loss_scale = 'dynamic' if name == 'mixed_float16' else None
    if loss_scale and self._compute_dtype not in (None, 'float16'):
      tf_logging.warn('Creating a Policy with a loss scale is only useful for '
                      'float16 policies. You passed loss_scale=%r for policy '
                      '%s. Consider not passing any loss_scale instead.' %
                      (loss_scale, name))
    self._loss_scale = loss_scale_module.get(loss_scale)

  def _parse_name(self, name):
    """Parses a Policy name into a compute and variable dtype.

    Args:
      name: The name of the policy:

    Returns:
      The (compute_dtype, variable_dtype) pair.
    """
    if name == 'mixed_float16':
      return 'float16', 'float32'
    elif name == 'mixed_bfloat16':
      return 'bfloat16', 'float32'

    if name.endswith('_with_float32_vars'):
      base_name = name[:-len('_with_float32_vars')]
      float32_vars = True
    else:
      base_name = name
      float32_vars = False

    if base_name == 'infer':
      base_dtype = None
    else:
      try:
        base_dtype = dtypes.as_dtype(base_name).name
      except TypeError:
        error = ('Cannot convert value %s to a mixed precision Policy. '
                 'Valid policies include include those in the form "<dtype>" '
                 'and "<dtype>_with_float32_vars", where <dtype> is the name '
                 'of a dtype.' % (name,))
        if float32_vars:
          error += (' The value %s ends with _with_float32_vars, but %s cannot '
                    'be converted to a DType' % (name, base_name))
        raise ValueError(error)

    if float32_vars:
      return base_dtype, 'float32'
    else:
      return base_dtype, base_dtype

  @property
  def variable_dtype(self):
    """The variable dtype of this policy.

    This is the dtype layers will create their variables in, unless a layer
    explicit chooses a different dtype. If this is different than
    `Policy.compute_dtype` and both are non-None, Layers will cast variables to
    the compute dtype to avoid type errors.

    If this is None, the policy is "infer" and the `compute_dtype` is also None.
    If `compute_dtype` is None, this is either None or float32.

    Returns:
      The variable dtype of this policy, or None if the variable dtype should be
      inferred from the inputs.
    """
    return self._variable_dtype

  @property
  def compute_dtype(self):
    """The compute dtype of this policy.

    This is the dtype layers will do their computations in.

    If this is None, the policy is "infer" or "infer_with_float32_vars" and
    `variable_dtype` is either None or float32 respectively.

    Note that even if the compute dtype is float16 or bfloat16, hardware devices
    may not do individual adds, multiplies, and other fundamental operations in
    [b]float16, but instead may do some of them in float32 for numeric
    stability. The compute dtype is the dtype of the inputs and outputs of the
    TensorFlow ops that the layer executes. Internally, many TensorFlow ops will
    do certain internal calculations in float32, or some other device-internal
    intermediate format with higher precision than [b]float16, to increase
    numeric stability.

    For example, a `tf.keras.layers.Dense` layer, when run on a GPU with a
    float16 compute dtype, will pass float16 inputs to tf.matmul. But, tf.matmul
    will do use float32 intermediate math. The performance benefit of float16 is
    still apparent, due to increased memory bandwidth and the fact GPUs have
    specialized hardware for computating matmuls on float16 while still keeping
    intermediate computations in float32.

    Returns:
      The variable dtype of this policy, or None if the variable dtype should be
      inferred from the inputs.
    """
    return self._compute_dtype

  @property
  def should_cast_variables(self):
    """Returns True if variables should be casted.

    This is true if the variable dtype is not the same as the compute dtype.

    Returns:
      True, if variables should be casted.
    """
    return self.variable_dtype != self.compute_dtype

  @property
  def loss_scale(self):
    """Returns the loss scale of this Policy.

    Returns:
      A `tf.train.experimental.LossScale`, or None.
    """
    return self._loss_scale

  @property
  def name(self):
    """Returns the name of this policy."""
    return self._name

  def __repr__(self):
    return '<Policy "%s", loss_scale=%s>' % (self._name, self.loss_scale)


def with_input_dtype(policy, dtype):
  """Copies "infer" `policy`, adding `dtype` to it.

  Policy must be "infer" or "infer_float32_vars" (i.e., has no compute dtype).
  Returns a new policy with compute dtype `dtype`. The returned policy's
  variable dtype is also `dtype` if `policy` is "infer", and is `float32` if
  `policy` is "infer_with_float32_vars".

  Args:
    policy: An "infer" or "infer_float32_vars" policy
    dtype: The dtype of an input to a layer.

  Returns:
    A new policy copied from `policy`, but with compute dtype and maybe
    variable_dtype set to `dtype`.
  """
  assert not policy.compute_dtype
  dtype = dtypes.as_dtype(dtype).name
  if policy.variable_dtype is None:
    return Policy(dtype)
  else:
    # Policies without a compute dtype are either "infer" or
    # "infer_with_float32_vars", so the variable_dtype must be float32 here.
    assert policy.variable_dtype == 'float32'
    return Policy(dtype + '_with_float32_vars')


# The current global policy in effect. If None, it means the current value of
# floatx should be used as the policy if the V2 dtype behavior is enabled,
# or "infer" otherwise.
# TODO(reedwm): Make this thread local?
_global_policy = None


@keras_export('keras.mixed_precision.experimental.global_policy')
def global_policy():
  """Returns the global Policy.

  The global policy is the default policy used for layers, if no policy is
  passed to the layer constructor. If no policy has been set with
  `keras.mixed_precision.experimental.set_policy`, this will return a policy
  constructed from `tf.keras.backend.floatx()` in TensorFlow 2, or an "infer"
  policy in TensorFlow 1.

  See `keras.mixed_precision.experimental.Policy` for more information.

  Returns:
    The global Policy.
  """
  if _global_policy is None:
    if base_layer_utils.v2_dtype_behavior_enabled():
      return Policy(backend.floatx())
    else:
      return Policy('infer')
  return _global_policy


def policy_defaults_to_floatx():
  """Returns True if `global_policy()` will use the current value of floatx."""
  return _global_policy is None and base_layer_utils.v2_dtype_behavior_enabled()


def _check_if_mixed_precision_graph_rewrite_is_enabled():
  # TODO(reedwm): Update this comment once the Keras API is complete.
  if mixed_precision_global_state.mixed_precision_graph_rewrite_is_enabled:
    raise ValueError(
        'The mixed precision policy cannot be set, because the mixed '
        'precision graph rewrite has already been enabled.\n'
        'At most, one of the following functions can be called:\n\n'
        '  1. tf.train.experimental.enable_mixed_precision_graph_rewrite() '
        '(You called this first)\n'
        '  2. tf.keras.mixed_precision.experimental.set_policy() (You called '
        'this second)\n\n'
        'You called both functions, which is an error, because both functions '
        'enable you to use mixed precision. The first function enables mixed '
        'precision in the graph with a graph rewrite. However it is currently '
        'not very customizable, and does not support eager. The second '
        'function is for Keras layers, but is not yet fully complete.')


@keras_export('keras.mixed_precision.experimental.set_policy')
def set_policy(policy):
  """Sets the global Policy.

  The global policy is the default policy used for layers, if no policy is
  passed to the layer constructor. If no global policy is set, layers will
  instead default to a Policy constructed from `tf.keras.backend.floatx()` in
  TensorFlow 2. In TensorFlow 1, layers default to an "infer" policy.

  See `keras.mixed_precision.experimental.Policy` for more information.

  Args:
    policy: A Policy, or a string that will be converted to a Policy..
  """
  global _global_policy
  _check_if_mixed_precision_graph_rewrite_is_enabled()
  if policy is not None and not isinstance(policy, Policy):
    policy = Policy(policy)
  if (policy and not base_layer_utils.v2_dtype_behavior_enabled() and
      policy.compute_dtype):
    raise ValueError(
        'The global policy can only be set to a non-infer policy in TensorFlow '
        '2')
  _global_policy = policy
  mixed_precision_global_state.using_default_mixed_precision_policy = (
      _global_policy is None)


# TODO(reedwm): Make this thread local
@contextlib.contextmanager
def policy_scope(policy):
  """A context manager that sets the global Policy under it.

  Args:
    policy: A Policy, or a string that will be converted to a Policy..

  Yields:
    Nothing.
  """
  old_policy = _global_policy
  try:
    set_policy(policy)
    yield
  finally:
    set_policy(old_policy)
