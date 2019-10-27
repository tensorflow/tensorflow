# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Built-in regularizers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.regularizers.Regularizer')
class Regularizer(object):
  """Regularizer base class.

  Regularizers allow you to apply penalties on layer parameters or layer
  activity during optimization. These penalties are summed into the loss
  function that the network optimizes.

  Regularization penalties are applied on a per-layer basis. The exact API will
  depend on the layer, but many layers (e.g. `Dense`, `Conv1D`, `Conv2D` and
  `Conv3D`) have a unified API.

  These layers expose 3 keyword arguments:

  - `kernel_regularizer`: Regularizer to apply a penalty on the layer's kernel
  - `bias_regularizer`: Regularizer to apply a penalty on the layer's bias
  - `activity_regularizer`: Regularizer to apply a penalty on the layer's output

  All layers (including custom layers) expose `activity_regularizer` as a
  settable property, whether or not it is in the constructor arguments.

  The value returned by the `activity_regularizer` is divided by the input
  batch size so that the relative weighting between the weight regularizers and
  the activity regularizers does not change with the batch size.

  You can access a layer's regularization penalties by calling `layer.losses`
  after calling the layer on inputs.

  ## Example

  >>> layer = tf.keras.layers.Dense(
  ...     5, input_dim=5,
  ...     kernel_initializer='ones',
  ...     kernel_regularizer=tf.keras.regularizers.l1(0.01),
  ...     activity_regularizer=tf.keras.regularizers.l2(0.01))
  >>> tensor = tf.ones(shape=(5, 5)) * 2.0
  >>> out = layer(tensor)

  >>> # The kernel regularization term is 0.25
  >>> # The activity regularization term (after dividing by the batch size) is 5
  >>> tf.math.reduce_sum(layer.losses)
  <tf.Tensor: shape=(), dtype=float32, numpy=5.25>

  ## Available penalties

  ```python
  tf.keras.regularizers.l1(0.3)  # L1 Regularization Penalty
  tf.keras.regularizers.l2(0.1)  # L2 Regularization Penalty
  tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)  # L1 + L2 penalties
  ```

  ## Directly calling a regularizer

  Compute a regularization loss on a tensor by directly calling a regularizer
  as if it is a one-argument function.

  E.g.
  >>> regularizer = tf.keras.regularizers.l2(2.)
  >>> tensor = tf.ones(shape=(5, 5))
  >>> regularizer(tensor)
  <tf.Tensor: shape=(), dtype=float32, numpy=50.0>


  ## Developing new regularizers

  Any function that takes in a weight matrix and returns a scalar
  tensor can be used as a regularizer, e.g.:

  >>> @tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
  ... def l1_reg(weight_matrix):
  ...    return 0.01 * tf.math.reduce_sum(tf.math.abs(weight_matrix))
  ...
  >>> layer = tf.keras.layers.Dense(5, input_dim=5,
  ...     kernel_initializer='ones', kernel_regularizer=l1_reg)
  >>> tensor = tf.ones(shape=(5, 5))
  >>> out = layer(tensor)
  >>> layer.losses
  [<tf.Tensor: shape=(), dtype=float32, numpy=0.25>]

  Alternatively, you can write your custom regularizers in an
  object-oriented way by extending this regularizer base class, e.g.:

  >>> @tf.keras.utils.register_keras_serializable(package='Custom', name='l2')
  ... class L2Regularizer(tf.keras.regularizers.Regularizer):
  ...   def __init__(self, l2=0.):  # pylint: disable=redefined-outer-name
  ...     self.l2 = l2
  ...
  ...   def __call__(self, x):
  ...     return self.l2 * tf.math.reduce_sum(tf.math.square(x))
  ...
  ...   def get_config(self):
  ...     return {'l2': float(self.l2)}
  ...
  >>> layer = tf.keras.layers.Dense(
  ...   5, input_dim=5, kernel_initializer='ones',
  ...   kernel_regularizer=L2Regularizer(l2=0.5))

  >>> tensor = tf.ones(shape=(5, 5))
  >>> out = layer(tensor)
  >>> layer.losses
  [<tf.Tensor: shape=(), dtype=float32, numpy=12.5>]

  ### A note on serialization and deserialization:

  Registering the regularizers as serializable is optional if you are just
  training and executing models, exporting to and from SavedModels, or saving
  and loading weight checkpoints.

  Registration is required for Keras `model_to_estimator`, saving and
  loading models to HDF5 formats, Keras model cloning, some visualization
  utilities, and exporting models to and from JSON. If using this functionality,
  you must make sure any python process running your model has also defined
  and registered your custom regularizer.

  `tf.keras.utils.register_keras_serializable` is only available in TF 2.1 and
  beyond. In earlier versions of TensorFlow you must pass your custom
  regularizer to the `custom_objects` argument of methods that expect custom
  regularizers to be registered as serializable.
  """

  def __call__(self, x):
    """Compute a regularization penalty from an input tensor."""
    return 0.

  @classmethod
  def from_config(cls, config):
    """Creates a regularizer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same regularizer from the config
    dictionary.

    This method is used by Keras `model_to_estimator`, saving and
    loading models to HDF5 formats, Keras model cloning, some visualization
    utilities, and exporting models to and from JSON.

    Arguments:
        config: A Python dictionary, typically the output of get_config.

    Returns:
        A regularizer instance.
    """
    return cls(**config)

  def get_config(self):
    """Returns the config of the regularizer.

    An regularizer config is a Python dictionary (serializable)
    containing all configuration parameters of the regularizer.
    The same regularizer can be reinstantiated later
    (without any saved state) from this configuration.

    This method is optional if you are just training and executing models,
    exporting to and from SavedModels, or using weight checkpoints.

    This method is required for Keras `model_to_estimator`, saving and
    loading models to HDF5 formats, Keras model cloning, some visualization
    utilities, and exporting models to and from JSON.

    Returns:
        Python dictionary.
    """
    raise NotImplementedError(str(self) + ' does not implement get_config()')


@keras_export('keras.regularizers.L1L2')
class L1L2(Regularizer):
  r"""A regularizer that applies both L1 and L2 regularization penalties.

  The L1 regularization penalty is computed as:
  $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

  The L2 regularization penalty is computed as
  $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$

  Attributes:
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.
  """

  def __init__(self, l1=0., l2=0.):  # pylint: disable=redefined-outer-name
    self.l1 = K.cast_to_floatx(l1)
    self.l2 = K.cast_to_floatx(l2)

  def __call__(self, x):
    if not self.l1 and not self.l2:
      return K.constant(0.)
    regularization = 0.
    if self.l1:
      regularization += self.l1 * math_ops.reduce_sum(math_ops.abs(x))
    if self.l2:
      regularization += self.l2 * math_ops.reduce_sum(math_ops.square(x))
    return regularization

  def get_config(self):
    return {'l1': float(self.l1), 'l2': float(self.l2)}


# Aliases.


@keras_export('keras.regularizers.l1')
def l1(l=0.01):
  r"""Create a regularizer that applies an L1 regularization penalty.

  The L1 regularization penalty is computed as:
  $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

  Arguments:
      l: Float; L1 regularization factor.

  Returns:
    An L1 Regularizer with the given regularization factor.
  """
  return L1L2(l1=l)


@keras_export('keras.regularizers.l2')
def l2(l=0.01):
  r"""Create a regularizer that applies an L2 regularization penalty.

  The L2 regularization penalty is computed as:
  $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$

  Arguments:
      l: Float; L2 regularization factor.

  Returns:
    An L2 Regularizer with the given regularization factor.
  """
  return L1L2(l2=l)


@keras_export('keras.regularizers.l1_l2')
def l1_l2(l1=0.01, l2=0.01):  # pylint: disable=redefined-outer-name
  r"""Create a regularizer that applies both L1 and L2 penalties.

  The L1 regularization penalty is computed as:
  $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

  The L2 regularization penalty is computed as:
  $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$

  Arguments:
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.

  Returns:
    An L1L2 Regularizer with the given regularization factors.
  """
  return L1L2(l1=l1, l2=l2)


@keras_export('keras.regularizers.serialize')
def serialize(regularizer):
  return serialize_keras_object(regularizer)


@keras_export('keras.regularizers.deserialize')
def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='regularizer')


@keras_export('keras.regularizers.get')
def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    identifier = str(identifier)
    # We have to special-case functions that return classes.
    # TODO(omalleyt): Turn these into classes or class aliases.
    special_cases = ['l1', 'l2', 'l1_l2']
    if identifier in special_cases:
      # Treat like a class.
      return deserialize({'class_name': identifier, 'config': {}})
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret regularizer identifier:', identifier)
