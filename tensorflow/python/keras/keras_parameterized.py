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

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class TestCase(test.TestCase, parameterized.TestCase):

  def tearDown(self):
    keras.backend.clear_session()
    super(TestCase, self).tearDown()


# TODO(kaftan): Possibly enable 'subclass_custom_build' when tests begin to pass
# it. Or perhaps make 'subclass' always use a custom build method.
def run_with_all_model_types(
    func=None,
    exclude_models=None,
    extra_parameters=None):
  """Execute the decorated test with all Keras model types.

  This decorator is intended to be applied to test methods in
  a `testing_utils.KerasTestCase` class. Doing so will cause the contents of
  the test method to be executed multiple times - once for each Keras model
  type.

  The Keras model types are: ['functional', 'subclass', 'sequential']

  Note: This decorator does not stack with any other parameterizing
  decorators. Use the extra_parameters arg if you would like to use additional
  parameterization.

  Various methods in `testing_utils` to get models will auto-generate a model
  of the currently active Keras model type. This allows unittests to confirm
  the equivalence between different Keras models.

  For example, consider the following unittest:

  ```python
  class MyTests(testing_utils.KerasTestCase):

    @testing_utils.run_with_all_model_types(
      model_types = ['functional', 'subclass'])
    def test_foo(self):
      model = testing_utils.get_small_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics)

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test tries building a small mlp as both a functional model and as a
  subclass model.


  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.
    exclude_models: A collection of Keras model types to not run.
      (May also be a single model type not wrapped in a collection).
      Defaults to None.
    extra_parameters: A list of dicts containing additional parameters for
      parameterizing the decorated test method. This additional parameters must
      be arguments in the test method's signature.
      See the absl.testing.parameterized decorators for more info on how these
      work.

  Returns:
    Returns a decorator that will run the decorated test method multiple times:
    once for each desired Keras model type.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
  return run_all_keras_modes_with_all_model_types(
      func, skip_run_modes=True, exclude_models=exclude_models,
      extra_parameters=extra_parameters)


def run_all_keras_modes(
    func=None,
    config=None,
    always_skip_v1=False,
    extra_parameters=None):
  """Execute the decorated test with all keras execution modes.

  This decorator is intended to be applied to test methods in
  a `testing_utils.KerasTestCase` class. Doing so will cause the contents of
  the test method to be executed three times - once executing in legacy graph
  mode, once running eagerly and with `should_run_eagerly` returning True, and
  once running eagerly with `should_run_eagerly` returning False.

  If Tensorflow v2 behavior is enabled, legacy graph mode will be skipped, and
  the test will only run twice.

  Note: This decorator does not stack with any other parameterizing
  decorators. Use the extra_parameters arg if you would like to use additional
  parameterization.

  For example, consider the following unittest:

  ```python
  class MyTests(testing_utils.KerasTestCase):

    @testing_utils.run_all_keras_modes
    def test_foo(self):
      model = testing_utils.get_small_functional_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics,
                    run_eagerly=testing_utils.should_run_eagerly())

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test will try compiling & fitting the small functional mlp using all
  three Keras execution modes.

  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.
    config: An optional config_pb2.ConfigProto to use to configure the
      session when executing graphs.
    always_skip_v1: If True, does not try running the legacy graph mode even
      when Tensorflow v2 behavior is not enabled.
    extra_parameters: A list of dicts containing additional parameters for
      parameterizing the decorated test method. This additional parameters must
      be arguments in the test method's signature. See the
      absl.testing.parameterized  decorators for more info on how these work.

  Returns:
    Returns a decorator that will run the decorated test method multiple times.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
  return run_all_keras_modes_with_all_model_types(
      func=func,
      skip_model_types=True,
      config=config,
      always_skip_v1=always_skip_v1,
      extra_parameters=extra_parameters)


def run_all_keras_modes_with_all_model_types(
    func=None,
    skip_run_modes=False,
    skip_model_types=False,
    config=None,
    always_skip_v1=False,
    exclude_models=None,
    extra_parameters=None):
  """Execute the decorated test with all keras run modes and model types.

  This decorator is intended to be applied to test methods in
  a `testing_utils.KerasTestCase` class. Doing so will cause the contents of
  the test method to be executed once for each combination of Keras run mode
  and model type.

  Note: This decorator does not stack with any other parameterizing
  decorators. Use the extra_parameters arg if you would like to use additional
  parameterization.

  For example, consider the following unittest:

  ```python
  class MyTests(testing_utils.KerasTestCase):

    @testing_utils.run_all_keras_modes_with_all_model_types
    def test_foo(self):
      model = testing_utils.get_small_functional_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics,
                    run_eagerly=testing_utils.should_run_eagerly())

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test will try compiling & fitting the small functional mlp using all
  combinations of the Keras execution modes and the Keras model types.

  The three Keras execution modes are v1 graph mode, v2 functions, and v2 eager.
  If Tensorflow v2 behavior is enabled, v1 graph mode will be skipped.
  The Keras model types are: ['functional', 'subclass', 'sequential']

  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.
    skip_run_modes: If True, does not try testing different Keras run modes.
      Defaults to False, which tests all combinations of run modes.
    skip_model_types: If True, does not try testing different Keras model types.
      Defaults to False, which tests all combinations of model types.
    config: An optional config_pb2.ConfigProto to use to configure the
      session when executing graphs.
    always_skip_v1: If True, does not try running the legacy graph mode even
      when Tensorflow v2 behavior is not enabled.
    exclude_models: A collection of Keras model types to not run.
      (May also be a single model type not wrapped in a collection).
    extra_parameters: A list of dicts containing additional parameters for
      parameterizing the decorated test method. This additional parameters must
      be arguments in the test method's signature. See the
      absl.testing.parameterized decorators for more info on how these work.

  Returns:
    Returns a decorator that will run the decorated test method multiple times.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
  if not parameterized:
    raise ImportError('To use the Keras parameterized testing utilities, you '
                      'must have absl.testing.parameterized installed and '
                      'included as a target dependency.')
  params = []
  if not skip_run_modes:
    if not (always_skip_v1 or tf2.enabled()):
      params.append({'testcase_name': '_v1_graph', '_run_mode': 'v1_graph'})
    params.append({'testcase_name': '_v2_eager', '_run_mode': 'v2_eager'})
    params.append({'testcase_name': '_v2_function', '_run_mode': 'v2_function'})
  else:
    params.append({'testcase_name': ''})

  if not skip_model_types:
    model_types = ['functional', 'subclass', 'sequential']
    old_params = params
    params = []
    for param in old_params:
      for model in model_types:
        if model not in nest.flatten(exclude_models):
          new_param = param.copy()
          new_param['_model_type'] = model
          new_param['testcase_name'] = '{}_{}'.format(
              param['testcase_name'], model)
          params.append(new_param)

  if extra_parameters:
    old_params = params
    params = []
    for param in old_params:
      for i, extra_params in enumerate(extra_parameters):
        new_param = param.copy()
        new_param.update(extra_params)
        new_param['testcase_name'] = '{}_{}'.format(
            param['testcase_name'], i)
        params.append(new_param)

  def decorator(f):
    """Decorator that constructs the test cases."""
    def run_with_model_type(self, model_type, *args, **kwargs):
      if model_type:
        with testing_utils.model_type_scope(model_type):
          f(self, *args, **kwargs)
      else:
        f(self, *args, **kwargs)

    # Use named_parameters so it can be individually run from the command line
    @parameterized.named_parameters(
        *params
    )
    def decorated(self, *args, **kwargs):
      """A run of a single test case w/ specified run mode / model type."""
      run_mode = kwargs.pop('_run_mode', None)
      model_type = kwargs.pop('_model_type', None)
      if run_mode is None:
        run_with_model_type(self, model_type, *args, **kwargs)
      elif run_mode == 'v1_graph':
        with context.graph_mode(), testing_utils.run_eagerly_scope(False):
          with self.test_session(use_gpu=True, config=config):
            run_with_model_type(self, model_type, *args, **kwargs)
      elif run_mode == 'v2_function':
        with context.eager_mode():
          with testing_utils.run_eagerly_scope(False):
            run_with_model_type(self, model_type, *args, **kwargs)
      elif run_mode == 'v2_eager':
        with context.eager_mode():
          with testing_utils.run_eagerly_scope(True):
            run_with_model_type(self, model_type, *args, **kwargs)
      else:
        return ValueError('Unknown run mode %s' % run_mode)

    return decorated

  if func is not None:
    return decorator(func)

  return decorator
