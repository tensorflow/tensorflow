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
from tensorflow.python.keras.mixed_precision.experimental import device_compatibility_check
from tensorflow.python.keras.mixed_precision.experimental import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util.tf_export import keras_export


# Default value of certain arguments, indicating the default behavior for
# that argument should be used.
USE_DEFAULT = 'USE_DEFAULT'


@keras_export('keras.mixed_precision.experimental.Policy', v1=[])
class Policy(object):
  """A dtype policy for a Keras layer.

  A dtype policy determines dtype-related aspects of a layer, such as its
  computation and variable dtypes. Each layer has a policy. Policies can be
  passed to the `dtype` argument of layer constructors, or a global policy can
  be set with `tf.keras.mixed_precision.experimental.set_policy`. A layer will
  default to the global policy if no policy is passed to it's constructor.

  For many models, each layer's policy will have the same compute dtype and
  variable dtype, which will typically be float32. In this case, we refer to the
  singular dtype as the layer's dtype, which can be queried by the property
  `tf.keras.layers.Layer.dtype`.

  When mixed precision training is used, most layers will instead have a float16
  or bfloat16 compute dtype and a float32 variable dtype, and so the layer does
  not have a single dtype. When the variable dtype does not match the compute
  dtype, variables will be automatically casted to the compute dtype to avoid
  type errors. In this case, `tf.keras.layers.Layer.dtype` refers to the
  variable dtype, not the compute dtype. See [the mixed precision guide](
    https://www.tensorflow.org/guide/keras/mixed_precision) for more
  information on how to use mixed precision.

  Certain policies also have a `tf.mixed_precision.experimental.LossScale`
  instance, which is used by `tf.keras.Model`s to performance loss scaling. Loss
  scaling is a technique used with mixed precision to avoid numerical underflow
  in float16 gradients. Loss scaling is only done by Models in `Model.fit`,
  `Model.train_on_batch`, and similar methods. Layers which are not Models
  ignore the loss scale.

  Policies are constructed by passing a string to the constructor, e.g.
  `tf.keras.mixed_precision.experimental.Policy('float32')`. The string
  determines the compute and variable dtypes. It can be one of the following:

    * Any dtype name, such as 'float32' or 'float64'. Both the variable and
      compute dtypes will be that dtype. No loss scaling is done by default.
    * 'mixed_float16' or 'mixed_bfloat16': The compute dtype is float16 or
      bfloat16, while the variable dtype is float32. These policies are used for
      mixed precision training. With 'mixed_float16', a dynamic loss scale is
      used by default. 'mixed_bfloat16' does no loss scaling by default, as loss
      scaling is unnecessary with bfloat16.

  ### How to use mixed precision in a Keras model

  To use mixed precision in a Keras model, the `'mixed_float16'` or
  `'mixed_bfloat16'` policy can be used.
  `tf.keras.mixed_precision.experimental.set_policy` can be used to set the
  default policy for layers if no policy is passed to them. For example:

  >>> tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
  >>> model = tf.keras.models.Sequential([
  ...     tf.keras.layers.Input((100,)),
  ...     # Dense layers use global policy of 'mixed_float16', which does
  ...     # computations in float16 while keeping variables in float32.
  ...     tf.keras.layers.Dense(10),
  ...     tf.keras.layers.Dense(10),
  ...     # Softmax should be done in float32 for numeric stability. We pass
  ...     # dtype='float32' to use float32 instead of the global policy.
  ...     tf.keras.layers.Activation('softmax', dtype='float32')
  ... ])

  Alternatively, the policy can be passed to individual layers instead of
  setting the global policy with `set_policy`:

  >>> policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
  >>> model = tf.keras.models.Sequential([
  ...     tf.keras.layers.Input((100,)),
  ...     tf.keras.layers.Dense(10, dtype=policy),
  ...     tf.keras.layers.Dense(10, dtype=policy),
  ...     # Softmax should be done in float32 for numeric stability.
  ...     tf.keras.layers.Activation('softmax', dtype='float32')
  ... ])

  Note the `'mixed_float16'` policy will apply loss scaling by default in
  `Model.fit`, `Model.train_on_batch`, and other training methods. If no such
  method is used (e.g., a custom training loop is used) and `'mixed_float16'` is
  used, the loss scale must be manually applied. See
  `tf.keras.mixed_precision.experimental.LossScaleOptimizer` for details. For
  `'mixed_bfloat16'`, no loss scaling is done and loss scaling never needs to be
  manually applied.

  See [the mixed precision guide](
    https://www.tensorflow.org/guide/keras/mixed_precision) for more
  information on using mixed precision

  ### How to use float64 in a Keras model

  Using float64 is similar to mixed precision. Either the global policy can be
  set to float64, or `dtype='float64'` can be passed to individual layers. For
  example, to set the global policy:

  >>> tf.keras.mixed_precision.experimental.set_policy('float64')
  >>> model = tf.keras.models.Sequential([
  ...     tf.keras.layers.Input((100,)),
  ...     # All layers use global policy of 'float64', which does computations
  ...     # and creates variables in float64.
  ...     tf.keras.layers.Dense(10),
  ...     tf.keras.layers.Dense(10),
  ...     tf.keras.layers.Activation('softmax')
  ... ])
  >>> # Optionaly set policy back to float32 if any other models use float32
  >>> tf.keras.mixed_precision.experimental.set_policy('float32')

  ### How a layer uses its policy's compute dtype

  A layer will cast its inputs to its compute dtype in TensorFlow 2. For
  example:

  >>> x = tf.ones((4, 4, 4, 4), dtype='float64')
  >>> # `layer`'s policy defaults to float32.
  >>> layer = tf.keras.layers.Conv2D(filters=4, kernel_size=2)
  >>> # `layer` casts it's inputs to its compute dtype, which is float32, and
  >>> # does computations in float32.
  >>> y = layer(x)
  >>> y.dtype
  tf.float32

  Note that the base `tf.keras.layers.Layer` class inserts the casts. If
  subclassing your own layer, you do not have to insert any casts.

  Currently, only tensors in the first argument to the layer's `call` method are
  casted. For example:

  >>> class MyLayer(tf.keras.layers.Layer):
  ...   # Bug! `b` will not be casted.
  ...   def call(self, a, b):
  ...     return a + 1., b + 1.
  >>> a = tf.constant(1., dtype="float32")
  >>> b = tf.constant(1., dtype="float32")
  >>> layer = MyLayer(dtype="float64")
  >>> x, y = layer(a, b)
  >>> x.dtype
  tf.float64
  >>> y.dtype
  tf.float32

  If writing your own layer, it is recommended to accept tensors only in the
  first argument. This way, all tensors are casted to the layer's compute dtype.
  `MyLayer` should therefore be written as:

  >>> class MyLayer(tf.keras.layers.Layer):
  ...   # Now, all tensor inputs will be casted.
  ...   def call(self, inputs):
  ...     a, b = inputs
  ...     return a + 1., b + 1.
  >>> a = tf.constant(1., dtype="float32")
  >>> b = tf.constant(1., dtype="float32")
  >>> layer = MyLayer(dtype="float64")
  >>> x, y = layer((a, b))
  >>> x.dtype
  tf.float64
  >>> y.dtype
  tf.float64

  Other arguments are not automatically casted for technical reasons, but this
  may change in a future minor release.

  A layer subclass can prevent its inputs from being autocasted by passing
  `autocast=False` to the layer constructor. For example:

  >>> class NonAutoCastingLayer(tf.keras.layers.Layer):
  ...   def __init__(self, **kwargs):
  ...     kwargs['autocast'] = False
  ...     super(NonAutoCastingLayer, self).__init__(**kwargs)
  ...   def call(self, inp):
  ...     return inp
  >>> x = tf.ones((4, 4, 4, 4), dtype='float32')
  >>> layer = NonAutoCastingLayer(dtype='float64')
  >>> y = layer(x)  # Will not cast inputs to it's compute dtype of float64
  >>> y.dtype
  tf.float32

  ### How a layer uses its policy's variable dtype

  The default dtype of variables created by `tf.keras.layers.Layer.add_weight`
  is the layer's policy's variable dtype.

  If a layer's compute and variable dtypes differ, `add_weight` will wrap
  floating-point variables with a special wrapper called an `AutoCastVariable`.
  This wrapper is identical to the original variable except it casts itself to
  the layer's compute dtype when used within `Layer.call`. Outside `Layer.call`,
  the variable is not casted.

  A layer author can prevent a variable from being wrapped with an
  `AutoCastVariable` by passing `experimental_autocast=False` to `add_weight`:

  >>> class MyLayer(tf.keras.layers.Layer):
  ...  def build(self, input_shape):
  ...    self.x = self.add_weight('x')
  ...    self.y = self.add_weight('y', experimental_autocast=False)
  >>> policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
  >>> layer = MyLayer(dtype=policy)
  >>> layer.build((2, 2))
  >>> layer.x
  <AutoCastVariable 'x:0' shape=() dtype=float32 true_dtype=float32, numpy=...>
  >>> layer.y
  <tf.Variable 'y:0' shape=() dtype=float32, numpy=...>

  Passing `experimental_autocast=False` is useful for layers which may
  internally do some math in the variable dtype instead of the compute dtype.
  For example, you may wish to compute variable statistics, such as mean and
  variance, in the variable dtype.

  ### How to write a layer that supports mixed precision and float64.

  For the most part, layers will automatically support mixed precision and
  float64 without any additional work, due to the fact the base layer
  automatically casts inputs, creates variables of the correct type, and in the
  case of mixed precision, wraps variables with `AutoCastVariables`.

  For example, this simple dense layer does not require any additional work to
  support mixed precision or float64. Keras automatically casts the inputs and
  variable to the appropriate dtype.

  >>> class MyDense(tf.keras.layers.Layer):
  ...   def build(self, input_shape):
  ...     self.kernel = self.add_weight('kernel', (input_shape[-1], 10))
  ...   def call(self, inputs):
  ...     return tf.matmul(inputs, self.kernel)

  >>> policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
  >>> layer = MyDense(dtype=policy)
  >>> x = np.random.rand(10, 10)
  >>> y = layer(x)
  >>> y.dtype
  tf.float16

  The primary case where you need extra work to support mixed precision or
  float64 is when you create a new tensor, such as with `tf.ones` or
  `tf.constant`. In such cases, you must create the tensor of the correct dtype.
  For example, suppose you modify the `MyDense` layer to add a random number to
  the output using `tf.random.normal`. You must pass the input dtype to
  `tf.random.normal` to ensure the dtypes match.

  >>> class MyDense(tf.keras.layers.Layer):
  ...   def build(self, input_shape):
  ...     self.kernel = self.add_weight('kernel', (input_shape[-1], 10))
  ...   def call(self, inputs):
  ...     rand = tf.random.normal(shape=inputs.shape, dtype=inputs.dtype)
  ...     return tf.matmul(inputs, self.kernel) + rand
  >>>
  >>> layer = MyDense(dtype=policy)
  >>> y = layer(x)
  >>> y.dtype
  tf.float16

  If you did not pass `dtype=inputs.dtype` to `tf.random.normal`, a `TypeError`
  would have occurred. This is because the dtype defaults to `"float32"`, so the
  layer would only work if the inputs were float32.
  """

  def __init__(self, name, loss_scale=USE_DEFAULT):
    """Constructs the policy.

    The `name` argument determines the compute and variable dtype, the default
    loss scale, and has no additional effect on the Policy. The compute and
    variable dtypes can only be specified through `name`, and cannot be
    specified directly.

    Args:
      name: A string. Can be one of the following values:
        * Any dtype name, such as 'float32' or 'float64'. Both the variable and
          compute dtypes will be that dtype.
        * 'mixed_float16' or 'mixed_bfloat16': The compute dtype is float16 or
          bfloat16, while the variable dtype is float32. With 'mixed_float16',
          a dynamic loss scale is used. These policies are used for mixed
          precision training.
      loss_scale: A `tf.mixed_precision.experimental.LossScale`, an int (which
        uses a `FixedLossScale`), or the string "dynamic" (which uses a
        `DynamicLossScale`). Defaults to using no loss scaling unless `name` is
        "mixed_float16", in which case this defaults to "dynamic". Only
        `tf.keras.Model`s, not layers, use the loss scale, and it is only used
        during `Model.fit`, `Model.train_on_batch`, and other similar methods.
    """
    if isinstance(name, dtypes.DType):
      raise TypeError("'name' must be a string, not a DType. "
                      "Instead, pass DType.name. Got: %s" % (name.name,))
    elif not isinstance(name, six.string_types):
      raise TypeError("'name' must be a string, but got: %s" % (name,))
    self._name = name
    self._compute_dtype, self._variable_dtype = self._parse_name(name)

    if loss_scale == USE_DEFAULT:
      loss_scale = 'dynamic' if name == 'mixed_float16' else None
      self._using_default_loss_scale = True
    else:
      self._using_default_loss_scale = False
    if loss_scale and self._compute_dtype not in (None, 'float16'):
      tf_logging.warn('Creating a Policy with a loss scale is only useful for '
                      'float16 policies. You passed loss_scale=%r for policy '
                      '%s. Consider not passing any loss_scale instead.' %
                      (loss_scale, name))
    self._loss_scale = keras_loss_scale_module.get(loss_scale)

    if name in ('mixed_float16', 'mixed_bloat16'):
      device_compatibility_check.log_device_compatibility_check(name,
                                                                skip_local=True)

  def _parse_name(self, name):
    """Parses a Policy name into a compute and variable dtype.

    Args:
      name: The name of the policy:

    Returns:
      The (compute_dtype, variable_dtype) pair.
    """
    if name.endswith('_float32_vars'):
      error_msg = ('Policies ending in \'_float32_vars\' have been removed '
                   'from TensorFlow.')
      if name in ('infer_float32_vars', 'infer_with_float32_vars'):
        error_msg += (' Please use the \'mixed_float16\' or \'mixed_bfloat16\' '
                      'policy instead.')
      elif name == 'float16_with_float32_vars':
        error_msg += (' Please use the \'mixed_float16\' policy instead.')
      elif name == 'bfloat16_with_float32_vars':
        error_msg += (' Please use the \'mixed_bfloat16\' policy instead.')
      error_msg += ' Got policy name: \'%s\'' % name
      raise ValueError(error_msg)

    if name == 'mixed_float16':
      return 'float16', 'float32'
    elif name == 'mixed_bfloat16':
      return 'bfloat16', 'float32'
    elif name == '_infer':
      # The "_infer" policy exists only for compatibility with TF 1, where
      # "_infer" is the default. The behavior matches the behavior of TF 1's
      # behavior before policies were introduced. With "_infer", the computation
      # and variable dtype are inferred from the first input the first time the
      # layer is called. Once the layer is called for the first time, the
      # layer's policy will change to the dtype of the first input, and it will
      # no longer have the "_infer" policy.
      #
      # The infer policy should be considered an implementation detail and may
      # be removed in the future.
      return None, None

    try:
      dtype = dtypes.as_dtype(name).name
    except TypeError:
      error = ("Cannot convert value %s to a mixed precision Policy. "
               "Valid policies include include 'mixed_float16', "
               "'mixed_bfloat16', and the name of any dtype such as "
               "'float32'." % (name,))
      # six.raise_from suppresses the original TypeError from being raised
      six.raise_from(ValueError(error), None)
    return dtype, dtype

  @property
  def variable_dtype(self):
    """The variable dtype of this policy.

    This is the dtype layers will create their variables in, unless a layer
    explicitly chooses a different dtype. If this is different than
    `Policy.compute_dtype`, Layers will cast variables to the compute dtype to
    avoid type errors.

    Returns:
      The variable dtype of this policy.
    """
    return self._variable_dtype

  @property
  def compute_dtype(self):
    """The compute dtype of this policy.

    This is the dtype layers will do their computations in.

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
    still apparent, due to increased memory bandwidth and the fact modern GPUs
    have specialized hardware for computing matmuls on float16 while still
    keeping intermediate computations in float32.

    Returns:
      The compute dtype of this policy.
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
      A `tf.mixed_precision.experimental.LossScale`, or None.
    """
    return self._loss_scale

  @property
  def name(self):
    """Returns the name of this policy."""
    return self._name

  def __repr__(self):
    return '<Policy "%s", loss_scale=%s>' % (self._name, self.loss_scale)

  def get_config(self):
    config = {
        'name': self.name
    }
    if not self._using_default_loss_scale:
      # We only include the loss scale if the default loss scale is not used.
      # This allows us to change the loss scale config format without breaking
      # users who use the default loss scale.
      config['loss_scale'] = keras_loss_scale_module.serialize(self.loss_scale)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'loss_scale' in config and isinstance(config['loss_scale'], dict):
      config = config.copy()
      config['loss_scale'] = keras_loss_scale_module.deserialize(
          config['loss_scale'], custom_objects=custom_objects)
    return cls(**config)


# The current global policy in effect. If None, it means the current value of
# floatx should be used as the policy if the V2 dtype behavior is enabled,
# or "_infer" otherwise.
# TODO(reedwm): Make this thread local?
_global_policy = None


@keras_export('keras.mixed_precision.experimental.global_policy', v1=[])
def global_policy():
  """Returns the global Policy.

  The global policy is the default policy used for layers, if no policy is
  passed to the layer constructor. If no policy has been set with
  `keras.mixed_precision.experimental.set_policy`, this will return a policy
  constructed from `tf.keras.backend.floatx()` (floatx defaults to float32).

  If TensorFlow 2 behavior has been disabled with
  `tf.compat.v1.disable_v2_behavior()`, this will instead return a special
  "_infer" policy which infers the dtype from the dtype of the first input the
  first time the layer is called. This behavior matches the behavior that
  existed in TensorFlow 1.

  See `tf.keras.mixed_precision.experimental.Policy` for more information on
  policies.

  Returns:
    The global Policy.
  """
  if _global_policy is None:
    if base_layer_utils.v2_dtype_behavior_enabled():
      return Policy(backend.floatx())
    else:
      return Policy('_infer')
  return _global_policy


def policy_defaults_to_floatx():
  """Returns True if `global_policy()` will use the current value of floatx."""
  return _global_policy is None and base_layer_utils.v2_dtype_behavior_enabled()


def _check_if_mixed_precision_graph_rewrite_is_enabled(policy):
  if mixed_precision_global_state.mixed_precision_graph_rewrite_is_enabled:
    raise ValueError(
        'The global dtype policy cannot be set to "{policy.name}", because the '
        'mixed precision graph rewrite has already been enabled.\n'
        'At most, one of the following can be called:\n\n'
        '  1. tf.train.experimental.enable_mixed_precision_graph_rewrite() '
        '(You called this first)\n'
        '  2. tf.keras.mixed_precision.experimental.set_policy() with a mixed '
        'precision policy (You called this second)\n\n'
        'You called both functions, which is an error, because both functions '
        'enable you to use mixed precision. If in doubt which function to use, '
        'use the second, as it supports Eager execution and is more '
        'customizable.'.format(policy=policy))


@keras_export('keras.mixed_precision.experimental.set_policy', v1=[])
def set_policy(policy):
  """Sets the global Policy.

  The global policy is the default policy used for layers, if no policy is
  passed to the layer constructor. If no global policy is set, layers will
  instead default to a Policy constructed from `tf.keras.backend.floatx()`.

  See `keras.mixed_precision.experimental.Policy` for more information.

  Args:
    policy: A Policy, or a string that will be converted to a Policy..
  """
  global _global_policy
  if not base_layer_utils.v2_dtype_behavior_enabled():
    raise ValueError('The global policy can only be set in TensorFlow 2')
  if policy is not None and not isinstance(policy, Policy):
    policy = Policy(policy)
  is_mixed_policy = policy is not None and policy.should_cast_variables
  if is_mixed_policy:
    _check_if_mixed_precision_graph_rewrite_is_enabled(policy)
  _global_policy = policy
  mixed_precision_global_state.using_mixed_precision_policy = is_mixed_policy


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


def _is_convertible_to_dtype(dtype):
  try:
    dtypes.as_dtype(dtype)
    return True
  except TypeError:
    return False


def _policy_equivalent_to_dtype(policy):
  """Returns True if the Policy is equivalent to a single dtype.

  A policy is equivalent to a single dtype if the policy's compute and variable
  dtypes are the same and the policy does not cause the layer/model to have
  additional behavior, such as loss scaling.

  The "_infer" policy is considered equivalent to a single dtype.

  Args:
    policy: A Policy.

  Returns:
    True, if the policy is equivalent to a single dtype.
  """
  # We use type() instead of isinstance because a sublcass of Policy is never
  # equivalent to a dtype.
  return (type(policy) == Policy and  # pylint: disable=unidiomatic-typecheck
          list(policy.get_config().keys()) == ['name'] and
          (policy.name == '_infer' or _is_convertible_to_dtype(policy.name)))


def serialize(policy):
  if _policy_equivalent_to_dtype(policy):
    # We return either None or the policy name for compatibility with older
    # versions of Keras. If the policy name is returned, it is a dtype string
    # such as 'float32'.
    return None if policy.name == '_infer' else policy.name
  return generic_utils.serialize_keras_object(policy)


def deserialize(config, custom_objects=None):
  if isinstance(config, str) and _is_convertible_to_dtype(config):
    return Policy(config)
  if config is None:
    return Policy('_infer')
  module_objects = {'Policy': Policy}
  return generic_utils.deserialize_keras_object(
      config,
      module_objects=module_objects,
      custom_objects=custom_objects,
      printable_module_name='dtype policy')
