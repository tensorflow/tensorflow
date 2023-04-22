# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras testing_utils."""

import unittest

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class KerasParameterizedTest(keras_parameterized.TestCase):

  def test_run_with_all_model_types(self):
    model_types = []
    models = []

    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_with_all_model_types
      def testBody(self):
        model_types.append(testing_utils.get_model_type())
        models.append(testing_utils.get_small_mlp(1, 4, input_dim=3))

    e = ExampleTest()
    e.testBody_functional()
    e.testBody_subclass()
    e.testBody_sequential()

    self.assertLen(model_types, 3)
    self.assertAllEqual(model_types, [
        "functional",
        "subclass",
        "sequential"
    ])

    # Validate that the models are what they should be
    self.assertTrue(models[0]._is_graph_network)
    self.assertFalse(models[1]._is_graph_network)
    self.assertNotIsInstance(models[0], keras.models.Sequential)
    self.assertNotIsInstance(models[1], keras.models.Sequential)
    self.assertIsInstance(models[2], keras.models.Sequential)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(model_types, 6)

  def test_run_with_all_model_types_and_extra_params(self):
    model_types = []
    models = []

    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_with_all_model_types
      @parameterized.named_parameters(
          [dict(testcase_name="_0", with_brackets=True),
           dict(testcase_name="_1", with_brackets=False)])
      def testBody(self, with_brackets):
        with_brackets = "with_brackets" if with_brackets else "without_brackets"
        model_types.append((with_brackets, testing_utils.get_model_type()))
        models.append(testing_utils.get_small_mlp(1, 4, input_dim=3))

    e = ExampleTest()
    e.testBody_0_functional()
    e.testBody_0_subclass()
    e.testBody_0_sequential()
    e.testBody_1_functional()
    e.testBody_1_subclass()
    e.testBody_1_sequential()

    self.assertLen(model_types, 6)
    self.assertAllEqual(model_types, [
        ("with_brackets", "functional"),
        ("with_brackets", "subclass"),
        ("with_brackets", "sequential"),
        ("without_brackets", "functional"),
        ("without_brackets", "subclass"),
        ("without_brackets", "sequential"),
    ])

    # Validate that the models are what they should be
    self.assertTrue(models[0]._is_graph_network)
    self.assertFalse(models[1]._is_graph_network)
    self.assertNotIsInstance(models[0], keras.models.Sequential)
    self.assertNotIsInstance(models[1], keras.models.Sequential)
    self.assertIsInstance(models[2], keras.models.Sequential)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(model_types, 12)

  def test_run_with_all_model_types_exclude_one(self):
    model_types = []
    models = []

    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_with_all_model_types(exclude_models="sequential")
      def testBody(self):
        model_types.append(testing_utils.get_model_type())
        models.append(testing_utils.get_small_mlp(1, 4, input_dim=3))

    e = ExampleTest()
    if hasattr(e, "testBody_functional"):
      e.testBody_functional()
    if hasattr(e, "testBody_subclass"):
      e.testBody_subclass()
    if hasattr(e, "testBody_sequential"):
      e.testBody_sequential()

    self.assertLen(model_types, 2)
    self.assertAllEqual(model_types, [
        "functional",
        "subclass"
    ])

    # Validate that the models are what they should be
    self.assertTrue(models[0]._is_graph_network)
    self.assertFalse(models[1]._is_graph_network)
    self.assertNotIsInstance(models[0], keras.models.Sequential)
    self.assertNotIsInstance(models[1], keras.models.Sequential)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(model_types, 4)

  def test_run_with_all_model_types_exclude_multiple(self):
    model_types = []
    models = []

    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_with_all_model_types(
          exclude_models=["sequential", "functional"])
      def testBody(self):
        model_types.append(testing_utils.get_model_type())
        models.append(testing_utils.get_small_mlp(1, 4, input_dim=3))

    e = ExampleTest()
    if hasattr(e, "testBody_functional"):
      e.testBody_functional()
    if hasattr(e, "testBody_subclass"):
      e.testBody_subclass()
    if hasattr(e, "testBody_sequential"):
      e.testBody_sequential()

    self.assertLen(model_types, 1)
    self.assertAllEqual(model_types, [
        "subclass"
    ])

    # Validate that the models are what they should be
    self.assertFalse(models[0]._is_graph_network)
    self.assertNotIsInstance(models[0], keras.models.Sequential)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(model_types, 2)

  def test_run_all_keras_modes(self):
    l = []

    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_all_keras_modes()
      def testBody(self):
        mode = "eager" if context.executing_eagerly() else "graph"
        should_run_eagerly = testing_utils.should_run_eagerly()
        l.append((mode, should_run_eagerly))

    e = ExampleTest()
    if not tf2.enabled():
      e.testBody_v1_session()
    e.testBody_v2_eager()
    e.testBody_v2_function()

    if not tf2.enabled():
      self.assertLen(l, 3)
      self.assertAllEqual(l, [
          ("graph", False),
          ("eager", True),
          ("eager", False),
      ])

      ts = unittest.makeSuite(ExampleTest)
      res = unittest.TestResult()
      ts.run(res)
      self.assertLen(l, 6)
    else:
      self.assertLen(l, 2)
      self.assertAllEqual(l, [
          ("eager", True),
          ("eager", False),
      ])

      ts = unittest.makeSuite(ExampleTest)
      res = unittest.TestResult()
      ts.run(res)
      self.assertLen(l, 4)

  def test_run_all_keras_modes_extra_params(self):
    l = []

    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_all_keras_modes()
      @parameterized.named_parameters(
          [dict(testcase_name="_0", with_brackets=True),
           dict(testcase_name="_1", with_brackets=False)])
      def testBody(self, with_brackets):
        mode = "eager" if context.executing_eagerly() else "graph"
        with_brackets = "with_brackets" if with_brackets else "without_brackets"
        should_run_eagerly = testing_utils.should_run_eagerly()
        l.append((with_brackets, mode, should_run_eagerly))

    e = ExampleTest()
    if not tf2.enabled():
      e.testBody_0_v1_session()
      e.testBody_1_v1_session()

    e.testBody_0_v2_eager()
    e.testBody_0_v2_function()
    e.testBody_1_v2_eager()
    e.testBody_1_v2_function()

    expected_combinations = {
        ("with_brackets", "eager", True),
        ("with_brackets", "eager", False),
        ("without_brackets", "eager", True),
        ("without_brackets", "eager", False),
    }

    if not tf2.enabled():
      expected_combinations = expected_combinations.union({
          ("with_brackets", "graph", False),
          ("without_brackets", "graph", False),
      })

    self.assertLen(l, len(expected_combinations))
    self.assertEqual(set(l), expected_combinations)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(l, len(expected_combinations) * 2)

  def test_run_all_keras_modes_always_skip_v1(self):
    l = []

    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
      def testBody(self):
        mode = "eager" if context.executing_eagerly() else "graph"
        should_run_eagerly = testing_utils.should_run_eagerly()
        l.append((mode, should_run_eagerly))

    e = ExampleTest()
    if hasattr(e, "testBody_v1_session"):
      e.testBody_v1_session()
    if hasattr(e, "testBody_v2_eager"):
      e.testBody_v2_eager()
    if hasattr(e, "testBody_v2_function"):
      e.testBody_v2_function()

    self.assertLen(l, 2)
    self.assertEqual(
        set(l), {
            ("eager", True),
            ("eager", False),
        })

  def test_run_all_keras_modes_with_all_model_types(self):
    l = []

    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_with_all_model_types
      @keras_parameterized.run_all_keras_modes
      def testBody(self):
        mode = "eager" if context.executing_eagerly() else "graph"
        should_run_eagerly = testing_utils.should_run_eagerly()
        l.append((mode, should_run_eagerly, testing_utils.get_model_type()))

    e = ExampleTest()
    e.testBody_v2_eager_functional()
    e.testBody_v2_function_functional()
    e.testBody_v2_eager_sequential()
    e.testBody_v2_function_sequential()
    e.testBody_v2_eager_subclass()
    e.testBody_v2_function_subclass()

    if not tf2.enabled():
      e.testBody_v1_session_functional()
      e.testBody_v1_session_sequential()
      e.testBody_v1_session_subclass()

    expected_combinations = {
        ("eager", True, "functional"),
        ("eager", False, "functional"),
        ("eager", True, "sequential"),
        ("eager", False, "sequential"),
        ("eager", True, "subclass"),
        ("eager", False, "subclass"),
    }

    if not tf2.enabled():
      expected_combinations = expected_combinations.union({
          ("graph", False, "functional"),
          ("graph", False, "sequential"),
          ("graph", False, "subclass"),
      })

    self.assertLen(l, len(expected_combinations))
    self.assertEqual(set(l), expected_combinations)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(l, len(expected_combinations) * 2)

  def test_run_all_model_types_with_all_keras_modes(self):
    l = []

    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_all_keras_modes
      @keras_parameterized.run_with_all_model_types
      def testBody(self):
        mode = "eager" if context.executing_eagerly() else "graph"
        should_run_eagerly = testing_utils.should_run_eagerly()
        l.append((mode, should_run_eagerly, testing_utils.get_model_type()))

    e = ExampleTest()
    e.testBody_functional_v2_eager()
    e.testBody_functional_v2_function()
    e.testBody_sequential_v2_eager()
    e.testBody_sequential_v2_function()
    e.testBody_subclass_v2_eager()
    e.testBody_subclass_v2_function()

    if not tf2.enabled():
      e.testBody_functional_v1_session()
      e.testBody_sequential_v1_session()
      e.testBody_subclass_v1_session()

    expected_combinations = {
        ("eager", True, "functional"),
        ("eager", False, "functional"),
        ("eager", True, "sequential"),
        ("eager", False, "sequential"),
        ("eager", True, "subclass"),
        ("eager", False, "subclass"),
    }

    if not tf2.enabled():
      expected_combinations = expected_combinations.union({
          ("graph", False, "functional"),
          ("graph", False, "sequential"),
          ("graph", False, "subclass"),
      })

    self.assertLen(l, len(expected_combinations))
    self.assertEqual(set(l), expected_combinations)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(l, len(expected_combinations) * 2)

  def test_run_all_keras_modes_with_all_model_types_annotate_class(self):
    l = []

    @keras_parameterized.run_with_all_model_types
    @keras_parameterized.run_all_keras_modes
    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @parameterized.named_parameters(dict(testcase_name="_arg",
                                           arg=True))
      def testBody(self, arg):
        mode = "eager" if context.executing_eagerly() else "graph"
        should_run_eagerly = testing_utils.should_run_eagerly()
        l.append((mode, should_run_eagerly, testing_utils.get_model_type()))

    e = ExampleTest()
    e.testBody_arg_v2_eager_functional()
    e.testBody_arg_v2_function_functional()
    e.testBody_arg_v2_eager_sequential()
    e.testBody_arg_v2_function_sequential()
    e.testBody_arg_v2_eager_subclass()
    e.testBody_arg_v2_function_subclass()

    if not tf2.enabled():
      e.testBody_arg_v1_session_functional()
      e.testBody_arg_v1_session_sequential()
      e.testBody_arg_v1_session_subclass()

    expected_combinations = {
        ("eager", True, "functional"),
        ("eager", False, "functional"),
        ("eager", True, "sequential"),
        ("eager", False, "sequential"),
        ("eager", True, "subclass"),
        ("eager", False, "subclass"),
    }

    if not tf2.enabled():
      expected_combinations = expected_combinations.union({
          ("graph", False, "functional"),
          ("graph", False, "sequential"),
          ("graph", False, "subclass"),
      })

    self.assertLen(l, len(expected_combinations))
    self.assertEqual(set(l), expected_combinations)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(l, len(expected_combinations) * 2)

  def test_run_all_keras_modes_with_all_model_types_annotate_class_2(self):
    l = []

    @keras_parameterized.run_with_all_model_types
    class ExampleTest(keras_parameterized.TestCase):

      def runTest(self):
        pass

      @keras_parameterized.run_all_keras_modes
      @parameterized.named_parameters(dict(testcase_name="_arg",
                                           arg=True))
      def testBody(self, arg):
        mode = "eager" if context.executing_eagerly() else "graph"
        should_run_eagerly = testing_utils.should_run_eagerly()
        l.append((mode, should_run_eagerly, testing_utils.get_model_type()))

    e = ExampleTest()
    e.testBody_arg_v2_eager_functional()
    e.testBody_arg_v2_function_functional()
    e.testBody_arg_v2_eager_sequential()
    e.testBody_arg_v2_function_sequential()
    e.testBody_arg_v2_eager_subclass()
    e.testBody_arg_v2_function_subclass()

    if not tf2.enabled():
      e.testBody_arg_v1_session_functional()
      e.testBody_arg_v1_session_sequential()
      e.testBody_arg_v1_session_subclass()

    expected_combinations = {
        ("eager", True, "functional"),
        ("eager", False, "functional"),
        ("eager", True, "sequential"),
        ("eager", False, "sequential"),
        ("eager", True, "subclass"),
        ("eager", False, "subclass"),
    }

    if not tf2.enabled():
      expected_combinations = expected_combinations.union({
          ("graph", False, "functional"),
          ("graph", False, "sequential"),
          ("graph", False, "subclass"),
      })

    self.assertLen(l, len(expected_combinations))
    self.assertEqual(set(l), expected_combinations)

    ts = unittest.makeSuite(ExampleTest)
    res = unittest.TestResult()
    ts.run(res)

    self.assertLen(l, len(expected_combinations) * 2)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(dict(testcase_name="argument",
                                       arg=True))
  def test_run_all_keras_modes_extra_params_2(self, arg):
    self.assertEqual(arg, True)

  @keras_parameterized.run_with_all_model_types
  @parameterized.named_parameters(dict(testcase_name="argument",
                                       arg=True))
  def test_run_with_all_model_types_extra_params_2(self, arg):
    self.assertEqual(arg, True)

if __name__ == "__main__":
  test.main()
