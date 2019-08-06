# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for unit-testing Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import numpy as np

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect


def get_test_data(train_samples,
                  test_samples,
                  input_shape,
                  num_classes,
                  random_seed=None):
  """Generates test data to train a model on.

  Arguments:
    train_samples: Integer, how many training samples to generate.
    test_samples: Integer, how many test samples to generate.
    input_shape: Tuple of integers, shape of the inputs.
    num_classes: Integer, number of classes for the data and targets.
    random_seed: Integer, random seed used by numpy to generate data.

  Returns:
    A tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  if random_seed is not None:
    np.random.seed(random_seed)
  num_sample = train_samples + test_samples
  templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
  y = np.random.randint(0, num_classes, size=(num_sample,))
  x = np.zeros((num_sample,) + input_shape, dtype=np.float32)
  for i in range(num_sample):
    x[i] = templates[y[i]] + np.random.normal(loc=0, scale=1., size=input_shape)
  return ((x[:train_samples], y[:train_samples]),
          (x[train_samples:], y[train_samples:]))


@test_util.use_deterministic_cudnn
def layer_test(layer_cls, kwargs=None, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None,
               expected_output_dtype=None, expected_output_shape=None,
               validate_training=True, adapt_data=None):
  """Test routine for a layer with a single input and single output.

  Arguments:
    layer_cls: Layer class object.
    kwargs: Optional dictionary of keyword arguments for instantiating the
      layer.
    input_shape: Input shape tuple.
    input_dtype: Data type of the input data.
    input_data: Numpy array of input data.
    expected_output: Numpy array of the expected output.
    expected_output_dtype: Data type expected for the output.
    expected_output_shape: Shape tuple for the expected shape of the output.
    validate_training: Whether to attempt to validate training on this layer.
      This might be set to False for non-differentiable layers that output
      string or integer values.
    adapt_data: Optional data for an 'adapt' call. If None, adapt() will not
      be tested for this layer. This is only relevant for PreprocessingLayers.

  Returns:
    The output data (Numpy array) returned by the layer, for additional
    checks to be done by the calling code.

  Raises:
    ValueError: if `input_shape is None`.
  """
  if input_data is None:
    if input_shape is None:
      raise ValueError('input_shape is None')
    if not input_dtype:
      input_dtype = 'float32'
    input_data_shape = list(input_shape)
    for i, e in enumerate(input_data_shape):
      if e is None:
        input_data_shape[i] = np.random.randint(1, 4)
    input_data = 10 * np.random.random(input_data_shape)
    if input_dtype[:5] == 'float':
      input_data -= 0.5
    input_data = input_data.astype(input_dtype)
  elif input_shape is None:
    input_shape = input_data.shape
  if input_dtype is None:
    input_dtype = input_data.dtype
  if expected_output_dtype is None:
    expected_output_dtype = input_dtype

  # instantiation
  kwargs = kwargs or {}
  layer = layer_cls(**kwargs)

  # Test adapt, if data was passed.
  if adapt_data is not None:
    layer.adapt(adapt_data)

  # test get_weights , set_weights at layer level
  weights = layer.get_weights()
  layer.set_weights(weights)

  # test and instantiation from weights
  if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
    kwargs['weights'] = weights
    layer = layer_cls(**kwargs)

  # test in functional API
  x = keras.layers.Input(shape=input_shape[1:], dtype=input_dtype)
  y = layer(x)
  if keras.backend.dtype(y) != expected_output_dtype:
    raise AssertionError('When testing layer %s, for input %s, found output '
                         'dtype=%s but expected to find %s.\nFull kwargs: %s' %
                         (layer_cls.__name__,
                          x,
                          keras.backend.dtype(y),
                          expected_output_dtype,
                          kwargs))

  def assert_shapes_equal(expected, actual):
    """Asserts that the output shape from the layer matches the actual shape."""
    if len(expected) != len(actual):
      raise AssertionError(
          'When testing layer %s, for input %s, found output_shape='
          '%s but expected to find %s.\nFull kwargs: %s' %
          (layer_cls.__name__, x, actual, expected, kwargs))

    for expected_dim, actual_dim in zip(expected, actual):
      if isinstance(expected_dim, tensor_shape.Dimension):
        expected_dim = expected_dim.value
      if isinstance(actual_dim, tensor_shape.Dimension):
        actual_dim = actual_dim.value
      if expected_dim is not None and expected_dim != actual_dim:
        raise AssertionError(
            'When testing layer %s, for input %s, found output_shape='
            '%s but expected to find %s.\nFull kwargs: %s' %
            (layer_cls.__name__, x, actual, expected, kwargs))

  if expected_output_shape is not None:
    assert_shapes_equal(tensor_shape.TensorShape(expected_output_shape),
                        y.shape)

  # check shape inference
  model = keras.models.Model(x, y)
  computed_output_shape = tuple(
      layer.compute_output_shape(
          tensor_shape.TensorShape(input_shape)).as_list())
  computed_output_signature = layer.compute_output_signature(
      tensor_spec.TensorSpec(shape=input_shape, dtype=input_dtype))
  actual_output = model.predict(input_data)
  actual_output_shape = actual_output.shape
  assert_shapes_equal(computed_output_shape, actual_output_shape)
  assert_shapes_equal(computed_output_signature.shape, actual_output_shape)
  if computed_output_signature.dtype != actual_output.dtype:
    raise AssertionError(
        'When testing layer %s, for input %s, found output_dtype='
        '%s but expected to find %s.\nFull kwargs: %s' %
        (layer_cls.__name__, x, actual_output.dtype,
         computed_output_signature.dtype, kwargs))
  if expected_output is not None:
    np.testing.assert_allclose(actual_output, expected_output,
                               rtol=1e-3, atol=1e-6)

  # test serialization, weight setting at model level
  model_config = model.get_config()
  recovered_model = keras.models.Model.from_config(model_config)
  if model.weights:
    weights = model.get_weights()
    recovered_model.set_weights(weights)
    output = recovered_model.predict(input_data)
    np.testing.assert_allclose(output, actual_output, rtol=1e-3, atol=1e-6)

  # test training mode (e.g. useful for dropout tests)
  # Rebuild the model to avoid the graph being reused between predict() and
  # See b/120160788 for more details. This should be mitigated after 2.0.
  if validate_training:
    model = keras.models.Model(x, layer(x))
    if _thread_local_data.run_eagerly is not None:
      model.compile(
          'rmsprop',
          'mse',
          weighted_metrics=['acc'],
          run_eagerly=should_run_eagerly())
    else:
      model.compile('rmsprop', 'mse', weighted_metrics=['acc'])
    model.train_on_batch(input_data, actual_output)

  # test as first layer in Sequential API
  layer_config = layer.get_config()
  layer_config['batch_input_shape'] = input_shape
  layer = layer.__class__.from_config(layer_config)

  # Test adapt, if data was passed.
  if adapt_data is not None:
    layer.adapt(adapt_data)

  model = keras.models.Sequential()
  model.add(layer)
  actual_output = model.predict(input_data)
  actual_output_shape = actual_output.shape
  for expected_dim, actual_dim in zip(computed_output_shape,
                                      actual_output_shape):
    if expected_dim is not None:
      if expected_dim != actual_dim:
        raise AssertionError(
            'When testing layer %s **after deserialization**, '
            'for input %s, found output_shape='
            '%s but expected to find inferred shape %s.\nFull kwargs: %s' %
            (layer_cls.__name__,
             x,
             actual_output_shape,
             computed_output_shape,
             kwargs))
  if expected_output is not None:
    np.testing.assert_allclose(actual_output, expected_output,
                               rtol=1e-3, atol=1e-6)

  # test serialization, weight setting at model level
  model_config = model.get_config()
  recovered_model = keras.models.Sequential.from_config(model_config)
  if model.weights:
    weights = model.get_weights()
    recovered_model.set_weights(weights)
    output = recovered_model.predict(input_data)
    np.testing.assert_allclose(output, actual_output, rtol=1e-3, atol=1e-6)

  # for further checks in the caller function
  return actual_output


_thread_local_data = threading.local()
_thread_local_data.model_type = None
_thread_local_data.run_eagerly = None
_thread_local_data.experimental_run_tf_function = None


@tf_contextlib.contextmanager
def model_type_scope(value):
  """Provides a scope within which the model type to test is equal to `value`.

  The model type gets restored to its original value upon exiting the scope.

  Arguments:
     value: model type value

  Yields:
    The provided value.
  """
  previous_value = _thread_local_data.model_type
  try:
    _thread_local_data.model_type = value
    yield value
  finally:
    # Restore model type to initial value.
    _thread_local_data.model_type = previous_value


@tf_contextlib.contextmanager
def run_eagerly_scope(value):
  """Provides a scope within which we compile models to run eagerly or not.

  The boolean gets restored to its original value upon exiting the scope.

  Arguments:
     value: Bool specifying if we should run models eagerly in the active test.
     Should be True or False.

  Yields:
    The provided value.
  """
  previous_value = _thread_local_data.run_eagerly
  try:
    _thread_local_data.run_eagerly = value
    yield value
  finally:
    # Restore model type to initial value.
    _thread_local_data.run_eagerly = previous_value


def should_run_eagerly():
  """Returns whether the models we are testing should be run eagerly."""
  if _thread_local_data.run_eagerly is None:
    raise ValueError('Cannot call `should_run_eagerly()` outside of a '
                     '`run_eagerly_scope()` or `run_all_keras_modes` '
                     'decorator.')

  return _thread_local_data.run_eagerly and context.executing_eagerly()


@tf_contextlib.contextmanager
def experimental_run_tf_function_scope(value):
  """Provides a scope within which we compile models to run with distribution.

  The boolean gets restored to its original value upon exiting the scope.

  Arguments:
     value: Bool specifying if we should run models with default distribution
     in the active test. Should be True or False.

  Yields:
    The provided value.
  """
  previous_value = _thread_local_data.experimental_run_tf_function
  try:
    _thread_local_data.experimental_run_tf_function = value
    yield value
  finally:
    # Restore model type to initial value.
    _thread_local_data.experimental_run_tf_function = previous_value


def should_run_tf_function():
  """Returns whether the models we are testing should be run distributed."""
  if _thread_local_data.experimental_run_tf_function is None:
    raise ValueError(
        'Cannot call `should_run_tf_function()` outside of a '
        '`experimental_run_tf_function_scope()` or `run_all_keras_modes` '
        'decorator.')

  return (_thread_local_data.experimental_run_tf_function and
          context.executing_eagerly())


def get_model_type():
  """Gets the model type that should be tested."""
  if _thread_local_data.model_type is None:
    raise ValueError('Cannot call `get_model_type()` outside of a '
                     '`model_type_scope()` or `run_with_all_model_types` '
                     'decorator.')

  return _thread_local_data.model_type


def get_small_sequential_mlp(num_hidden, num_classes, input_dim=None):
  model = keras.models.Sequential()
  if input_dim:
    model.add(keras.layers.Dense(num_hidden, activation='relu',
                                 input_dim=input_dim))
  else:
    model.add(keras.layers.Dense(num_hidden, activation='relu'))
  activation = 'sigmoid' if num_classes == 1 else 'softmax'
  model.add(keras.layers.Dense(num_classes, activation=activation))
  return model


def get_small_functional_mlp(num_hidden, num_classes, input_dim):
  inputs = keras.Input(shape=(input_dim,))
  outputs = keras.layers.Dense(num_hidden, activation='relu')(inputs)
  activation = 'sigmoid' if num_classes == 1 else 'softmax'
  outputs = keras.layers.Dense(num_classes, activation=activation)(outputs)
  return keras.Model(inputs, outputs)


class _SmallSubclassMLP(keras.Model):
  """A subclass model based small MLP."""

  def __init__(self, num_hidden, num_classes):
    super(_SmallSubclassMLP, self).__init__()
    self.layer_a = keras.layers.Dense(num_hidden, activation='relu')
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    self.layer_b = keras.layers.Dense(num_classes, activation=activation)

  def call(self, inputs, **kwargs):
    x = self.layer_a(inputs)
    return self.layer_b(x)


class _SmallSubclassMLPCustomBuild(keras.Model):
  """A subclass model small MLP that uses a custom build method."""

  def __init__(self, num_hidden, num_classes):
    super(_SmallSubclassMLPCustomBuild, self).__init__()
    self.layer_a = None
    self.layer_b = None
    self.num_hidden = num_hidden
    self.num_classes = num_classes

  def build(self, input_shape):
    self.layer_a = keras.layers.Dense(self.num_hidden, activation='relu')
    activation = 'sigmoid' if self.num_classes == 1 else 'softmax'
    self.layer_b = keras.layers.Dense(self.num_classes, activation=activation)

  def call(self, inputs, **kwargs):
    x = self.layer_a(inputs)
    return self.layer_b(x)


def get_small_subclass_mlp(num_hidden, num_classes):
  return _SmallSubclassMLP(num_hidden, num_classes)


def get_small_subclass_mlp_with_custom_build(num_hidden, num_classes):
  return _SmallSubclassMLPCustomBuild(num_hidden, num_classes)


def get_small_mlp(num_hidden, num_classes, input_dim):
  """Get a small mlp of the model type specified by `get_model_type`."""
  model_type = get_model_type()
  if model_type == 'subclass':
    return get_small_subclass_mlp(num_hidden, num_classes)
  if model_type == 'subclass_custom_build':
    return get_small_subclass_mlp_with_custom_build(num_hidden, num_classes)
  if model_type == 'sequential':
    return get_small_sequential_mlp(num_hidden, num_classes, input_dim)
  if model_type == 'functional':
    return get_small_functional_mlp(num_hidden, num_classes, input_dim)
  raise ValueError('Unknown model type {}'.format(model_type))


class _SubclassModel(keras.Model):
  """A Keras subclass model."""

  def __init__(self, layers):
    super(_SubclassModel, self).__init__()
    # Note that clone and build doesn't support lists of layers in subclassed
    # models. Adding each layer directly here.
    for i, layer in enumerate(layers):
      setattr(self, self._layer_name_for_i(i), layer)

    self.num_layers = len(layers)

  def _layer_name_for_i(self, i):
    return 'layer{}'.format(i)

  def call(self, inputs, **kwargs):
    x = inputs
    for i in range(self.num_layers):
      layer = getattr(self, self._layer_name_for_i(i))
      x = layer(x)
    return x


class _SubclassModelCustomBuild(keras.Model):
  """A Keras subclass model that uses a custom build method."""

  def __init__(self, layer_generating_func):
    super(_SubclassModelCustomBuild, self).__init__()
    self.all_layers = None
    self._layer_generating_func = layer_generating_func

  def build(self, input_shape):
    layers = []
    for layer in self._layer_generating_func():
      layers.append(layer)
    self.all_layers = layers

  def call(self, inputs, **kwargs):
    x = inputs
    for layer in self.all_layers:
      x = layer(x)
    return x


def get_model_from_layers(layers, input_shape=None, input_dtype=None):
  """Builds a model from a sequence of layers."""
  model_type = get_model_type()
  if model_type == 'subclass':
    return _SubclassModel(layers)

  if model_type == 'subclass_custom_build':
    layer_generating_func = lambda: layers
    return _SubclassModelCustomBuild(layer_generating_func)

  if model_type == 'sequential':
    model = keras.models.Sequential()
    if input_shape:
      model.add(keras.layers.InputLayer(input_shape=input_shape,
                                        dtype=input_dtype))
    for layer in layers:
      model.add(layer)
    return model

  if model_type == 'functional':
    if not input_shape:
      raise ValueError('Cannot create a functional model from layers with no '
                       'input shape.')
    inputs = keras.Input(shape=input_shape, dtype=input_dtype)
    outputs = inputs
    for layer in layers:
      outputs = layer(outputs)
    return keras.Model(inputs, outputs)

  raise ValueError('Unknown model type {}'.format(model_type))


class _MultiIOSubclassModel(keras.Model):
  """Multi IO Keras subclass model."""

  def __init__(self, branch_a, branch_b, shared_input_branch=None,
               shared_output_branch=None):
    super(_MultiIOSubclassModel, self).__init__()
    self._shared_input_branch = shared_input_branch
    self._branch_a = branch_a
    self._branch_b = branch_b
    self._shared_output_branch = shared_output_branch

  def call(self, inputs, **kwargs):
    if self._shared_input_branch:
      for layer in self._shared_input_branch:
        inputs = layer(inputs)
      a = inputs
      b = inputs
    else:
      a, b = inputs

    for layer in self._branch_a:
      a = layer(a)
    for layer in self._branch_b:
      b = layer(b)
    outs = [a, b]

    if self._shared_output_branch:
      for layer in self._shared_output_branch:
        outs = layer(outs)

    return outs


class _MultiIOSubclassModelCustomBuild(keras.Model):
  """Multi IO Keras subclass model that uses a custom build method."""

  def __init__(self, branch_a_func, branch_b_func,
               shared_input_branch_func=None,
               shared_output_branch_func=None):
    super(_MultiIOSubclassModelCustomBuild, self).__init__()
    self._shared_input_branch_func = shared_input_branch_func
    self._branch_a_func = branch_a_func
    self._branch_b_func = branch_b_func
    self._shared_output_branch_func = shared_output_branch_func

    self._shared_input_branch = None
    self._branch_a = None
    self._branch_b = None
    self._shared_output_branch = None

  def build(self, input_shape):
    if self._shared_input_branch_func():
      self._shared_input_branch = self._shared_input_branch_func()
    self._branch_a = self._branch_a_func()
    self._branch_b = self._branch_b_func()

    if self._shared_output_branch_func():
      self._shared_output_branch = self._shared_output_branch_func()

  def call(self, inputs, **kwargs):
    if self._shared_input_branch:
      for layer in self._shared_input_branch:
        inputs = layer(inputs)
      a = inputs
      b = inputs
    else:
      a, b = inputs

    for layer in self._branch_a:
      a = layer(a)
    for layer in self._branch_b:
      b = layer(b)
    outs = a, b

    if self._shared_output_branch:
      for layer in self._shared_output_branch:
        outs = layer(outs)

    return outs


def get_multi_io_model(
    branch_a,
    branch_b,
    shared_input_branch=None,
    shared_output_branch=None):
  """Builds a multi-io model that contains two branches.

  The produced model will be of the type specified by `get_model_type`.

  To build a two-input, two-output model:
    Specify a list of layers for branch a and branch b, but do not specify any
    shared input branch or shared output branch. The resulting model will apply
    each branch to a different input, to produce two outputs.

    The first value in branch_a must be the Keras 'Input' layer for branch a,
    and the first value in branch_b must be the Keras 'Input' layer for
    branch b.

    example usage:
    ```
    branch_a = [Input(shape=(2,), name='a'), Dense(), Dense()]
    branch_b = [Input(shape=(3,), name='b'), Dense(), Dense()]

    model = get_multi_io_model(branch_a, branch_b)
    ```

  To build a two-input, one-output model:
    Specify a list of layers for branch a and branch b, and specify a
    shared output branch. The resulting model will apply
    each branch to a different input. It will then apply the shared output
    branch to a tuple containing the intermediate outputs of each branch,
    to produce a single output. The first layer in the shared_output_branch
    must be able to merge a tuple of two tensors.

    The first value in branch_a must be the Keras 'Input' layer for branch a,
    and the first value in branch_b must be the Keras 'Input' layer for
    branch b.

    example usage:
    ```
    input_branch_a = [Input(shape=(2,), name='a'), Dense(), Dense()]
    input_branch_b = [Input(shape=(3,), name='b'), Dense(), Dense()]
    shared_output_branch = [Concatenate(), Dense(), Dense()]

    model = get_multi_io_model(input_branch_a, input_branch_b,
                               shared_output_branch=shared_output_branch)
    ```
  To build a one-input, two-output model:
    Specify a list of layers for branch a and branch b, and specify a
    shared input branch. The resulting model will take one input, and apply
    the shared input branch to it. It will then respectively apply each branch
    to that intermediate result in parallel, to produce two outputs.

    The first value in the shared_input_branch must be the Keras 'Input' layer
    for the whole model. Branch a and branch b should not contain any Input
    layers.

    example usage:
    ```
    shared_input_branch = [Input(shape=(2,), name='in'), Dense(), Dense()]
    output_branch_a = [Dense(), Dense()]
    output_branch_b = [Dense(), Dense()]


    model = get_multi_io_model(output__branch_a, output_branch_b,
                               shared_input_branch=shared_input_branch)
    ```

  Args:
    branch_a: A sequence of layers for branch a of the model.
    branch_b: A sequence of layers for branch b of the model.
    shared_input_branch: An optional sequence of layers to apply to a single
      input, before applying both branches to that intermediate result. If set,
      the model will take only one input instead of two. Defaults to None.
    shared_output_branch: An optional sequence of layers to merge the
      intermediate results produced by branch a and branch b. If set,
      the model will produce only one output instead of two. Defaults to None.

  Returns:
    A multi-io model of the type specified by `get_model_type`, specified
    by the different branches.
  """
  # Extract the functional inputs from the layer lists
  if shared_input_branch:
    inputs = shared_input_branch[0]
    shared_input_branch = shared_input_branch[1:]
  else:
    inputs = branch_a[0], branch_b[0]
    branch_a = branch_a[1:]
    branch_b = branch_b[1:]

  model_type = get_model_type()
  if model_type == 'subclass':
    return _MultiIOSubclassModel(branch_a, branch_b, shared_input_branch,
                                 shared_output_branch)

  if model_type == 'subclass_custom_build':
    return _MultiIOSubclassModelCustomBuild((lambda: branch_a),
                                            (lambda: branch_b),
                                            (lambda: shared_input_branch),
                                            (lambda: shared_output_branch))

  if model_type == 'sequential':
    raise ValueError('Cannot use `get_multi_io_model` to construct '
                     'sequential models')

  if model_type == 'functional':
    if shared_input_branch:
      a_and_b = inputs
      for layer in shared_input_branch:
        a_and_b = layer(a_and_b)
      a = a_and_b
      b = a_and_b
    else:
      a, b = inputs

    for layer in branch_a:
      a = layer(a)
    for layer in branch_b:
      b = layer(b)
    outputs = a, b

    if shared_output_branch:
      for layer in shared_output_branch:
        outputs = layer(outputs)

    return keras.Model(inputs, outputs)

  raise ValueError('Unknown model type {}'.format(model_type))


_V2_OPTIMIZER_MAP = {
    'adadelta': adadelta_v2.Adadelta,
    'adagrad': adagrad_v2.Adagrad,
    'adam': adam_v2.Adam,
    'adamax': adamax_v2.Adamax,
    'nadam': nadam_v2.Nadam,
    'rmsprop': rmsprop_v2.RMSprop,
    'sgd': gradient_descent_v2.SGD
}


def get_v2_optimizer(name, **kwargs):
  """Get the v2 optimizer requested.

  This is only necessary until v2 are the default, as we are testing in Eager,
  and Eager + v1 optimizers fail tests. When we are in v2, the strings alone
  should be sufficient, and this mapping can theoretically be removed.

  Args:
    name: string name of Keras v2 optimizer.
    **kwargs: any kwargs to pass to the optimizer constructor.

  Returns:
    Initialized Keras v2 optimizer.

  Raises:
    ValueError: if an unknown name was passed.
  """
  try:
    return _V2_OPTIMIZER_MAP[name](**kwargs)
  except KeyError:
    raise ValueError(
        'Could not find requested v2 optimizer: {}\nValid choices: {}'.format(
            name, list(_V2_OPTIMIZER_MAP.keys())))


def get_expected_metric_variable_names(var_names, name_suffix=''):
  """Returns expected metric variable names given names and prefix/suffix."""
  if tf2.enabled() or context.executing_eagerly():
    # In V1 eager mode and V2 variable names are not made unique.
    return [n + ':0' for n in var_names]
  # In V1 graph mode variable names are made unique using a suffix.
  return [n + name_suffix + ':0' for n in var_names]


def enable_v2_dtype_behavior(fn):
  """Decorator for enabling the layer V2 dtype behavior on a test."""
  return _set_v2_dtype_behavior(fn, True)


def disable_v2_dtype_behavior(fn):
  """Decorator for disabling the layer V2 dtype behavior on a test."""
  return _set_v2_dtype_behavior(fn, False)


def _set_v2_dtype_behavior(fn, enabled):
  """Returns version of 'fn' that runs with v2 dtype behavior on or off."""
  def wrapper(*args, **kwargs):
    v2_dtype_behavior = base_layer_utils.V2_DTYPE_BEHAVIOR
    base_layer_utils.V2_DTYPE_BEHAVIOR = enabled
    try:
      return fn(*args, **kwargs)
    finally:
      base_layer_utils.V2_DTYPE_BEHAVIOR = v2_dtype_behavior

  return wrapper
