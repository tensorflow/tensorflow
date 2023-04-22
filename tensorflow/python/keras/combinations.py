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
"""This module customizes `test_combinations` for `tf.keras` related tests."""

import functools

from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras import testing_utils

KERAS_MODEL_TYPES = ['functional', 'subclass', 'sequential']


def keras_mode_combinations(mode=None, run_eagerly=None):
  """Returns the default test combinations for tf.keras tests.

  Note that if tf2 is enabled, then v1 session test will be skipped.

  Args:
    mode: List of modes to run the tests. The valid options are 'graph' and
      'eager'. Default to ['graph', 'eager'] if not specified. If a empty list
      is provide, then the test will run under the context based on tf's
      version, eg graph for v1 and eager for v2.
    run_eagerly: List of `run_eagerly` value to be run with the tests.
      Default to [True, False] if not specified. Note that for `graph` mode,
      run_eagerly value will only be False.

  Returns:
    A list contains all the combinations to be used to generate test cases.
  """
  if mode is None:
    mode = ['eager'] if tf2.enabled() else ['graph', 'eager']
  if run_eagerly is None:
    run_eagerly = [True, False]
  result = []
  if 'eager' in mode:
    result += combinations.combine(mode=['eager'], run_eagerly=run_eagerly)
  if 'graph' in mode:
    result += combinations.combine(mode=['graph'], run_eagerly=[False])
  return result


def keras_model_type_combinations():
  return combinations.combine(model_type=KERAS_MODEL_TYPES)


class KerasModeCombination(test_combinations.TestCombination):
  """Combination for Keras test mode.

  It by default includes v1_session, v2_eager and v2_tf_function.
  """

  def context_managers(self, kwargs):
    run_eagerly = kwargs.pop('run_eagerly', None)

    if run_eagerly is not None:
      return [testing_utils.run_eagerly_scope(run_eagerly)]
    else:
      return []

  def parameter_modifiers(self):
    return [test_combinations.OptionalParameter('run_eagerly')]


class KerasModelTypeCombination(test_combinations.TestCombination):
  """Combination for Keras model types when doing model test.

  It by default includes 'functional', 'subclass', 'sequential'.

  Various methods in `testing_utils` to get models will auto-generate a model
  of the currently active Keras model type. This allows unittests to confirm
  the equivalence between different Keras models.
  """

  def context_managers(self, kwargs):
    model_type = kwargs.pop('model_type', None)
    if model_type in KERAS_MODEL_TYPES:
      return [testing_utils.model_type_scope(model_type)]
    else:
      return []

  def parameter_modifiers(self):
    return [test_combinations.OptionalParameter('model_type')]


_defaults = combinations.generate.keywords['test_combinations']
generate = functools.partial(
    combinations.generate,
    test_combinations=_defaults +
    (KerasModeCombination(), KerasModelTypeCombination()))
combine = test_combinations.combine
times = test_combinations.times
NamedObject = test_combinations.NamedObject
