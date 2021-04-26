# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-classes-have-attributes
"""Python module for Keras base types.

All the classes in this module is abstract classes that contains none or minimal
implementations. It is designed be used as base class for other concrete
classes, type checks, and python3 type hints.
"""

import abc

# TODO(scottzhu): Export all the types under this module with API symbol.


class Layer(object, metaclass=abc.ABCMeta):
  """This is the class from which all layers inherit.

  A layer is a callable object that takes as input one or more tensors and
  that outputs one or more tensors. It involves *computation*, defined
  in the `call()` method, and a *state* (weight variables), defined
  either in the constructor `__init__()` or in the `build()` method.

  Users will just instantiate a layer and then treat it as a callable.

  We recommend that descendants of `Layer` implement the following methods:

  * `__init__()`: Defines custom layer attributes, and creates layer state
    variables that do not depend on input shapes, using `add_weight()`.
  * `build(self, input_shape)`: This method can be used to create weights that
    depend on the shape(s) of the input(s), using `add_weight()`. `__call__()`
    will automatically build the layer (if it has not been built yet) by
    calling `build()`.
  * `call(self, *args, **kwargs)`: Called in `__call__` after making sure
    `build()` has been called. `call()` performs the logic of applying the
    layer to the input tensors (which should be passed in as argument).
    Two reserved keyword arguments you can optionally use in `call()` are:
      - `training` (boolean, whether the call is in
        inference mode or training mode)
      - `mask` (boolean tensor encoding masked timesteps in the input, used
        in RNN layers)
  * `get_config(self)`: Returns a dictionary containing the configuration used
    to initialize this layer. If the keys differ from the arguments
    in `__init__`, then override `from_config(self)` as well.
    This method is used when saving
    the layer or a model that contains this layer.

  Examples:

  Here's a basic example: a layer with two variables, `w` and `b`,
  that returns `y = w . x + b`.
  It shows how to implement `build()` and `call()`.
  Variables set as attributes of a layer are tracked as weights
  of the layers (in `layer.weights`).

  ```python
  class SimpleDense(Layer):

    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
      w_init = tf.random_normal_initializer()
      self.w = tf.Variable(
          initial_value=w_init(shape=(input_shape[-1], self.units),
                               dtype='float32'),
          trainable=True)
      b_init = tf.zeros_initializer()
      self.b = tf.Variable(
          initial_value=b_init(shape=(self.units,), dtype='float32'),
          trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.matmul(inputs, self.w) + self.b

  # Instantiates the layer.
  linear_layer = SimpleDense(4)

  # This will also call `build(input_shape)` and create the weights.
  y = linear_layer(tf.ones((2, 2)))
  assert len(linear_layer.weights) == 2

  # These weights are trainable, so they're listed in `trainable_weights`:
  assert len(linear_layer.trainable_weights) == 2
  ```

  Note that the method `add_weight()` offers a shortcut to create weights:

  ```python
  class SimpleDense(Layer):

    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
  ```

  Besides trainable weights, updated via backpropagation during training,
  layers can also have non-trainable weights. These weights are meant to
  be updated manually during `call()`. Here's a example layer that computes
  the running sum of its inputs:

  ```python
  class ComputeSum(Layer):

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # Create a non-trainable weight.
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                                 trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total

  my_sum = ComputeSum(2)
  x = tf.ones((2, 2))

  y = my_sum(x)
  print(y.numpy())  # [2. 2.]

  y = my_sum(x)
  print(y.numpy())  # [4. 4.]

  assert my_sum.weights == [my_sum.total]
  assert my_sum.non_trainable_weights == [my_sum.total]
  assert my_sum.trainable_weights == []
  ```

  For more information about creating layers, see the guide
  [Making new Layers and Models via subclassing](
    https://www.tensorflow.org/guide/keras/custom_layers_and_models)

  Args:
    trainable: Boolean, whether the layer's variables should be trainable.
    name: String name of the layer.
    dtype: The dtype of the layer's computations and weights (default of
      `None` means use `tf.keras.backend.floatx` in TensorFlow 2, or the type
      of the first input in TensorFlow 1).
    dynamic: Set this to `True` if your layer should only be run eagerly, and
      should not be used to generate a static computation graph.
      This would be the case for a Tree-RNN or a recursive network,
      for example, or generally for any layer that manipulates tensors
      using Python control flow. If `False`, we assume that the layer can
      safely be used to generate a static computation graph.

  Attributes:
    name: The name of the layer (string).
    dtype: The dtype of the layer's computations and weights. If mixed
      precision is used with a `tf.keras.mixed_precision.Policy`, this is
      instead just the dtype of the layer's weights, as the computations are
      done in a different dtype.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer.
    trainable_weights: List of variables to be included in backprop.
    non_trainable_weights: List of variables that should not be
      included in backprop.
    weights: The concatenation of the lists trainable_weights and
      non_trainable_weights (in this order).
    trainable: Whether the layer should be trained (boolean).
    input_spec: Optional (list of) `InputSpec` object(s) specifying the
      constraints on inputs that can be accepted by the layer.

  Each layer has a dtype, which is typically the dtype of the layer's
  computations and variables. A layer's dtype can be queried via the
  `Layer.dtype` property. The dtype is specified with the `dtype` constructor
  argument. In TensorFlow 2, the dtype defaults to `tf.keras.backend.floatx()`
  if no dtype is passed. `floatx()` itself defaults to "float32". Additionally,
  layers will cast their inputs to the layer's dtype in TensorFlow 2. When mixed
  precision is used, layers may have different computation and variable dtypes.
  See `tf.keras.mixed_precision.Policy` for details on layer dtypes.
  """
  pass
