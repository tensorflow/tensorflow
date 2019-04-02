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

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.mixed_precision.experimental.Policy')
class Policy(object):
  """A mixed precision policy for a Keras layer.

  A mixed precision policy determines the floating-point dtype that Keras layers
  should create variables in. For non-default policies, if the variable dtype
  does not match the input dtype, variables will automatically be casted to the
  input dtype to avoid type errors. Policies can be passed to the 'dtype'
  argument of layer constructors, or a global policy can be set with
  'set_policy'.

  In the near future, policies will also determine the computation dtype of
  layers, as well as the loss scaling algorithm.

  Policies are intended to enable mixed precision training, which require using
  float32 variables and [b]float16 computations for most layers. The term "mixed
  precision" refers to the use of both float16 (or bfloat16) and float32 in a
  model. See https://arxiv.org/abs/1710.03740 for more information on mixed
  precision training.

  Policies are constructed by passing a string to the `name` constructor
  argument. `name` determines the behavior of the policy. Currently, `name` can
  be one of the following values.

    * 'infer': Infer the variable and computation dtypes from the input dtype.
      This is the default behavior.
    * 'infer_float32_vars': Infer the computation dtypes from the input
      dtype, but create variables in float32. Variables will be casted to the
      computation dtype. This is intended to enable mixed precision. Users can
      cast tensors to float16 before passing them to a layer, which causes the
      layer to run it's computation in float16 while keeping variables in
      float32.

  To use mixed precision in a model, the 'infer_float32_vars' policy can be used
  alongside float16 input tensors, which results in float16 computations and
  float32 variables. For example:

  ```python
  tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
  model = tf.keras.models.Sequential(
      tf.keras.layers.Input((100,), dtype='float16'),
      tf.keras.layers.Dense(10),
      tf.keras.layers.Dense(10),
      tf.keras.layers.Lambda(lambda x: tf.cast(x, 'float32')),
      tf.keras.layers.Activation('Softmax')
  )
  ```

  Alternatively, the policy can be passed to individual layers instead of
  setting the global policy with `set_policy`:

  ```python
  policy = tf.keras.mixed_precision.experimental.Policy('infer_float32_vars')
  model = tf.keras.models.Sequential(
      tf.keras.layers.Input((100,), dtype='float16'),
      tf.keras.layers.Dense(10, dtype=policy),
      tf.keras.layers.Dense(10, dtype=policy),
      tf.keras.layers.Lambda(lambda x: tf.cast(x, 'float32')),
      tf.keras.layers.Activation('Softmax')
  )
  ```

  Note that a LossScaleOptimizer should also be used for mixed precision models
  to avoid numerical underflow. See `LossScaleOptimizer`.
  """

  def __init__(self, name):
    self._name = name
    if name == 'infer':
      self._default_variable_dtype = None
    elif name == 'infer_float32_vars':
      self._default_variable_dtype = 'float32'
    else:
      raise ValueError('"name" argument to Policy constructor must be "infer" '
                       'or "infer_float32_vars", but got: %s' % name)

  @property
  def name(self):
    """Returns the name of the policy: "infer" or "infer_float32_vars."""
    return self._name

  @property
  def default_variable_dtype(self):
    """Returns the default variable dtype of this policy.

    This is the dtype layers will create their variables in, unless a layer
    explicit chooses a different dtype. Layers will cast variables to the
    appropriate dtype to avoid type errors.

    Returns:
      The default variable dtype of this policy, or None if the default variable
      dtype should be derived from the inputs.
    """
    return self._default_variable_dtype

  @property
  def should_cast_variables(self):
    """Returns true if variables should be casted."""
    return self.default_variable_dtype is not None

  # TODO(reedwm): Implement get_config/from_config.


# TODO(reedwm): Make this thread local?
_global_policy = Policy('infer')


@keras_export('keras.mixed_precision.experimental.global_policy')
def global_policy():
  """Returns the global Policy.

  The global policy is the default policy used for layers, if no policy is
  passed to the layer constructor. When TensorFlow starts, the global policy is
  set to an "infer" policy, and can be changed with `set_policy`.

  Returns:
    The global Policy.
  """
  return _global_policy


@keras_export('keras.mixed_precision.experimental.set_policy')
def set_policy(policy):
  """Sets the global Policy."""
  global _global_policy
  if not isinstance(policy, Policy):
    policy = Policy(policy)
  _global_policy = policy


# TODO(reedwm): Make this thread local
@contextlib.contextmanager
def policy_scope(policy):
  old_policy = _global_policy
  try:
    set_policy(policy)
    yield
  finally:
    set_policy(old_policy)
