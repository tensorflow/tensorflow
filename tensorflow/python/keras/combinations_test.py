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
"""Tests for Keras combinations."""

import unittest
from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.keras import combinations
from tensorflow.python.keras import models as keras_models
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class CombinationsTest(test.TestCase):

  def test_run_all_keras_modes(self):
    test_params = []

    class ExampleTest(parameterized.TestCase):

      def runTest(self):
        pass

      @combinations.generate(combinations.keras_mode_combinations())
      def testBody(self):
        mode = "eager" if context.executing_eagerly() else "graph"
        should_run_eagerly = testing_utils.should_run_eagerly()
        test_params.append((mode, should_run_eagerly))

    e = ExampleTest()
    if not tf2.enabled():
      e.testBody_test_mode_graph_runeagerly_False()
    e.testBody_test_mode_eager_runeagerly_True()
    e.testBody_test_mode_eager_runeagerly_False()

    if not tf2.enabled():
      self.assertLen(test_params, 3)
      self.assertAllEqual(test_params, [
          ("graph", False),
          ("eager", True),
          ("eager", False),
      ])

      ts = unittest.makeSuite(ExampleTest)
      res = unittest.TestResult()
      ts.run(res)
      self.assertLen(test_params, 6)
    else:
      self.assertLen(test_params, 2)
      self.assertAllEqual(test_params, [
          ("eager", True),
          ("eager", False),
      ])

      ts = unittest.makeSuite(ExampleTest)
      res = unittest.TestResult()
      ts.run(res)
      self.assertLen(test_params, 4)

  def test_generate_keras_mode_eager_only(self):
    result = combinations.keras_mode_combinations(mode=["eager"])
    self.assertLen(result, 2)
    self.assertEqual(result[0], {"mode": "eager", "run_eagerly": True})
    self.assertEqual(result[1], {"mode": "eager", "run_eagerly": False})

  def test_generate_keras_mode_skip_run_eagerly(self):
    result = combinations.keras_mode_combinations(run_eagerly=[False])
    if tf2.enabled():
      self.assertLen(result, 1)
      self.assertEqual(result[0], {"mode": "eager", "run_eagerly": False})
    else:
      self.assertLen(result, 2)
      self.assertEqual(result[0], {"mode": "eager", "run_eagerly": False})
      self.assertEqual(result[1], {"mode": "graph", "run_eagerly": False})

  def test_run_all_keras_model_types(self):
    model_types = []
    models = []

    class ExampleTest(parameterized.TestCase):

      def runTest(self):
        pass

      @combinations.generate(combinations.keras_model_type_combinations())
      def testBody(self):
        model_types.append(testing_utils.get_model_type())
        models.append(testing_utils.get_small_mlp(1, 4, input_dim=3))

    e = ExampleTest()
    e.testBody_test_modeltype_functional()
    e.testBody_test_modeltype_subclass()
    e.testBody_test_modeltype_sequential()

    self.assertLen(model_types, 3)
    self.assertAllEqual(model_types, [
        "functional",
        "subclass",
        "sequential"
    ])

    # Validate that the models are what they should be
    self.assertTrue(models[0]._is_graph_network)
    self.assertFalse(models[1]._is_graph_network)
    self.assertNotIsInstance(models[0], keras_models.Sequential)
    self.assertNotIsInstance(models[1], keras_models.Sequential)
    self.assertIsInstance(models[2], keras_models.Sequential)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(model_types, 6)

  def test_combine_combinations(self):
    test_cases = []

    @combinations.generate(combinations.times(
        combinations.keras_mode_combinations(),
        combinations.keras_model_type_combinations()))
    class ExampleTest(parameterized.TestCase):

      def runTest(self):
        pass

      @parameterized.named_parameters(dict(testcase_name="_arg",
                                           arg=True))
      def testBody(self, arg):
        del arg
        mode = "eager" if context.executing_eagerly() else "graph"
        should_run_eagerly = testing_utils.should_run_eagerly()
        test_cases.append((mode, should_run_eagerly,
                           testing_utils.get_model_type()))

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    expected_combinations = [
        ("eager", False, "functional"),
        ("eager", False, "sequential"),
        ("eager", False, "subclass"),
        ("eager", True, "functional"),
        ("eager", True, "sequential"),
        ("eager", True, "subclass"),
    ]

    if not tf2.enabled():
      expected_combinations.extend([
          ("graph", False, "functional"),
          ("graph", False, "sequential"),
          ("graph", False, "subclass"),
      ])

    self.assertAllEqual(sorted(test_cases), expected_combinations)


if __name__ == "__main__":
  test.main()
