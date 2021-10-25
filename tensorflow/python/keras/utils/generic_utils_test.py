# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Keras generic Python utils."""

from functools import partial

import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import test


class HasArgTest(test.TestCase):

  def test_has_arg(self):

    def f_x(x):
      return x

    def f_x_args(x, *args):
      _ = args
      return x

    def f_x_kwargs(x, **kwargs):
      _ = kwargs
      return x

    def f(a, b, c):
      return a + b + c

    partial_f = partial(f, b=1)

    self.assertTrue(keras.utils.generic_utils.has_arg(
        f_x, 'x', accept_all=False))
    self.assertFalse(keras.utils.generic_utils.has_arg(
        f_x, 'y', accept_all=False))
    self.assertTrue(keras.utils.generic_utils.has_arg(
        f_x_args, 'x', accept_all=False))
    self.assertFalse(keras.utils.generic_utils.has_arg(
        f_x_args, 'y', accept_all=False))
    self.assertTrue(keras.utils.generic_utils.has_arg(
        f_x_kwargs, 'x', accept_all=False))
    self.assertFalse(keras.utils.generic_utils.has_arg(
        f_x_kwargs, 'y', accept_all=False))
    self.assertTrue(keras.utils.generic_utils.has_arg(
        f_x_kwargs, 'y', accept_all=True))
    self.assertTrue(
        keras.utils.generic_utils.has_arg(partial_f, 'c', accept_all=True))


class TestCustomObjectScope(test.TestCase):

  def test_custom_object_scope(self):

    def custom_fn():
      pass

    class CustomClass(object):
      pass

    with keras.utils.generic_utils.custom_object_scope(
        {'CustomClass': CustomClass, 'custom_fn': custom_fn}):
      act = keras.activations.get('custom_fn')
      self.assertEqual(act, custom_fn)
      cl = keras.regularizers.get('CustomClass')
      self.assertEqual(cl.__class__, CustomClass)


class SerializeKerasObjectTest(test.TestCase):

  def test_serialize_none(self):
    serialized = keras.utils.generic_utils.serialize_keras_object(None)
    self.assertEqual(serialized, None)
    deserialized = keras.utils.generic_utils.deserialize_keras_object(
        serialized)
    self.assertEqual(deserialized, None)

  def test_serialize_custom_class_with_default_name(self):

    @keras.utils.generic_utils.register_keras_serializable()
    class TestClass(object):

      def __init__(self, value):
        self._value = value

      def get_config(self):
        return {'value': self._value}

    serialized_name = 'Custom>TestClass'
    inst = TestClass(value=10)
    class_name = keras.utils.generic_utils._GLOBAL_CUSTOM_NAMES[TestClass]
    self.assertEqual(serialized_name, class_name)
    config = keras.utils.generic_utils.serialize_keras_object(inst)
    self.assertEqual(class_name, config['class_name'])
    new_inst = keras.utils.generic_utils.deserialize_keras_object(config)
    self.assertIsNot(inst, new_inst)
    self.assertIsInstance(new_inst, TestClass)
    self.assertEqual(10, new_inst._value)

    # Make sure registering a new class with same name will fail.
    with self.assertRaisesRegex(ValueError, '.*has already been registered.*'):
      @keras.utils.generic_utils.register_keras_serializable()  # pylint: disable=function-redefined
      class TestClass(object):  # pylint: disable=function-redefined

        def __init__(self, value):
          self._value = value

        def get_config(self):
          return {'value': self._value}

  def test_serialize_custom_class_with_custom_name(self):

    @keras.utils.generic_utils.register_keras_serializable(
        'TestPackage', 'CustomName')
    class OtherTestClass(object):

      def __init__(self, val):
        self._val = val

      def get_config(self):
        return {'val': self._val}

    serialized_name = 'TestPackage>CustomName'
    inst = OtherTestClass(val=5)
    class_name = keras.utils.generic_utils._GLOBAL_CUSTOM_NAMES[OtherTestClass]
    self.assertEqual(serialized_name, class_name)
    fn_class_name = keras.utils.generic_utils.get_registered_name(
        OtherTestClass)
    self.assertEqual(fn_class_name, class_name)

    cls = keras.utils.generic_utils.get_registered_object(fn_class_name)
    self.assertEqual(OtherTestClass, cls)

    config = keras.utils.generic_utils.serialize_keras_object(inst)
    self.assertEqual(class_name, config['class_name'])
    new_inst = keras.utils.generic_utils.deserialize_keras_object(config)
    self.assertIsNot(inst, new_inst)
    self.assertIsInstance(new_inst, OtherTestClass)
    self.assertEqual(5, new_inst._val)

  def test_serialize_custom_function(self):

    @keras.utils.generic_utils.register_keras_serializable()
    def my_fn():
      return 42

    serialized_name = 'Custom>my_fn'
    class_name = keras.utils.generic_utils._GLOBAL_CUSTOM_NAMES[my_fn]
    self.assertEqual(serialized_name, class_name)
    fn_class_name = keras.utils.generic_utils.get_registered_name(my_fn)
    self.assertEqual(fn_class_name, class_name)

    config = keras.utils.generic_utils.serialize_keras_object(my_fn)
    self.assertEqual(class_name, config)
    fn = keras.utils.generic_utils.deserialize_keras_object(config)
    self.assertEqual(42, fn())

    fn_2 = keras.utils.generic_utils.get_registered_object(fn_class_name)
    self.assertEqual(42, fn_2())

  def test_serialize_custom_class_without_get_config_fails(self):

    with self.assertRaisesRegex(
        ValueError, 'Cannot register a class that does '
        'not have a get_config.*'):

      @keras.utils.generic_utils.register_keras_serializable(  # pylint: disable=unused-variable
          'TestPackage', 'TestClass')
      class TestClass(object):

        def __init__(self, value):
          self._value = value

  def test_serializable_object(self):

    class SerializableInt(int):
      """A serializable object to pass out of a test layer's config."""

      def __new__(cls, value):
        return int.__new__(cls, value)

      def get_config(self):
        return {'value': int(self)}

      @classmethod
      def from_config(cls, config):
        return cls(**config)

    layer = keras.layers.Dense(
        SerializableInt(3),
        activation='relu',
        kernel_initializer='ones',
        bias_regularizer='l2')
    config = keras.layers.serialize(layer)
    new_layer = keras.layers.deserialize(
        config, custom_objects={'SerializableInt': SerializableInt})
    self.assertEqual(new_layer.activation, keras.activations.relu)
    self.assertEqual(new_layer.bias_regularizer.__class__,
                     keras.regularizers.L2)
    self.assertEqual(new_layer.units.__class__, SerializableInt)
    self.assertEqual(new_layer.units, 3)

  def test_nested_serializable_object(self):
    class SerializableInt(int):
      """A serializable object to pass out of a test layer's config."""

      def __new__(cls, value):
        return int.__new__(cls, value)

      def get_config(self):
        return {'value': int(self)}

      @classmethod
      def from_config(cls, config):
        return cls(**config)

    class SerializableNestedInt(int):
      """A serializable object containing another serializable object."""

      def __new__(cls, value, int_obj):
        obj = int.__new__(cls, value)
        obj.int_obj = int_obj
        return obj

      def get_config(self):
        return {'value': int(self), 'int_obj': self.int_obj}

      @classmethod
      def from_config(cls, config):
        return cls(**config)

    nested_int = SerializableInt(4)
    layer = keras.layers.Dense(
        SerializableNestedInt(3, nested_int),
        name='SerializableNestedInt',
        activation='relu',
        kernel_initializer='ones',
        bias_regularizer='l2')
    config = keras.layers.serialize(layer)
    new_layer = keras.layers.deserialize(
        config,
        custom_objects={
            'SerializableInt': SerializableInt,
            'SerializableNestedInt': SerializableNestedInt
        })
    # Make sure the string field doesn't get convert to custom object, even
    # they have same value.
    self.assertEqual(new_layer.name, 'SerializableNestedInt')
    self.assertEqual(new_layer.activation, keras.activations.relu)
    self.assertEqual(new_layer.bias_regularizer.__class__,
                     keras.regularizers.L2)
    self.assertEqual(new_layer.units.__class__, SerializableNestedInt)
    self.assertEqual(new_layer.units, 3)
    self.assertEqual(new_layer.units.int_obj.__class__, SerializableInt)
    self.assertEqual(new_layer.units.int_obj, 4)

  def test_nested_serializable_fn(self):

    def serializable_fn(x):
      """A serializable function to pass out of a test layer's config."""
      return x

    class SerializableNestedInt(int):
      """A serializable object containing a serializable function."""

      def __new__(cls, value, fn):
        obj = int.__new__(cls, value)
        obj.fn = fn
        return obj

      def get_config(self):
        return {'value': int(self), 'fn': self.fn}

      @classmethod
      def from_config(cls, config):
        return cls(**config)

    layer = keras.layers.Dense(
        SerializableNestedInt(3, serializable_fn),
        activation='relu',
        kernel_initializer='ones',
        bias_regularizer='l2')
    config = keras.layers.serialize(layer)
    new_layer = keras.layers.deserialize(
        config,
        custom_objects={
            'serializable_fn': serializable_fn,
            'SerializableNestedInt': SerializableNestedInt
        })
    self.assertEqual(new_layer.activation, keras.activations.relu)
    self.assertIsInstance(new_layer.bias_regularizer, keras.regularizers.L2)
    self.assertIsInstance(new_layer.units, SerializableNestedInt)
    self.assertEqual(new_layer.units, 3)
    self.assertIs(new_layer.units.fn, serializable_fn)

  def test_serialize_type_object_initializer(self):
    layer = keras.layers.Dense(
        1,
        kernel_initializer=keras.initializers.ones,
        bias_initializer=keras.initializers.zeros)
    config = keras.layers.serialize(layer)
    self.assertEqual(config['config']['bias_initializer']['class_name'],
                     'Zeros')
    self.assertEqual(config['config']['kernel_initializer']['class_name'],
                     'Ones')

  def test_serializable_with_old_config(self):
    # model config generated by tf-1.2.1
    old_model_config = {
        'class_name':
            'Sequential',
        'config': [{
            'class_name': 'Dense',
            'config': {
                'name': 'dense_1',
                'trainable': True,
                'batch_input_shape': [None, 784],
                'dtype': 'float32',
                'units': 32,
                'activation': 'linear',
                'use_bias': True,
                'kernel_initializer': {
                    'class_name': 'Ones',
                    'config': {
                        'dtype': 'float32'
                    }
                },
                'bias_initializer': {
                    'class_name': 'Zeros',
                    'config': {
                        'dtype': 'float32'
                    }
                },
                'kernel_regularizer': None,
                'bias_regularizer': None,
                'activity_regularizer': None,
                'kernel_constraint': None,
                'bias_constraint': None
            }
        }]
    }
    old_model = keras.utils.generic_utils.deserialize_keras_object(
        old_model_config, module_objects={'Sequential': keras.Sequential})
    new_model = keras.Sequential([
        keras.layers.Dense(32, input_dim=784, kernel_initializer='Ones'),
    ])
    input_data = np.random.normal(2, 1, (5, 784))
    output = old_model.predict(input_data)
    expected_output = new_model.predict(input_data)
    self.assertAllEqual(output, expected_output)

  def test_deserialize_unknown_object(self):

    class CustomLayer(keras.layers.Layer):
      pass

    layer = CustomLayer()
    config = keras.utils.generic_utils.serialize_keras_object(layer)
    with self.assertRaisesRegexp(ValueError,
                                 'passed to the `custom_objects` arg'):
      keras.utils.generic_utils.deserialize_keras_object(config)
    restored = keras.utils.generic_utils.deserialize_keras_object(
        config, custom_objects={'CustomLayer': CustomLayer})
    self.assertIsInstance(restored, CustomLayer)


class SliceArraysTest(test.TestCase):

  def test_slice_arrays(self):
    input_a = list([1, 2, 3])
    self.assertEqual(
        keras.utils.generic_utils.slice_arrays(input_a, start=0),
        [None, None, None])
    self.assertEqual(
        keras.utils.generic_utils.slice_arrays(input_a, stop=3),
        [None, None, None])
    self.assertEqual(
        keras.utils.generic_utils.slice_arrays(input_a, start=0, stop=1),
        [None, None, None])


# object() alone isn't compatible with WeakKeyDictionary, which we use to
# track shared configs.
class MaybeSharedObject(object):
  pass


class SharedObjectScopeTest(test.TestCase):

  def test_shared_object_saving_scope_single_object_doesnt_export_id(self):
    with generic_utils.SharedObjectSavingScope() as scope:
      single_object = MaybeSharedObject()
      self.assertIsNone(scope.get_config(single_object))
      single_object_config = scope.create_config({}, single_object)
      self.assertIsNotNone(single_object_config)
      self.assertNotIn(generic_utils.SHARED_OBJECT_KEY,
                       single_object_config)

  def test_shared_object_saving_scope_shared_object_exports_id(self):
    with generic_utils.SharedObjectSavingScope() as scope:
      shared_object = MaybeSharedObject()
      self.assertIsNone(scope.get_config(shared_object))
      scope.create_config({}, shared_object)
      first_object_config = scope.get_config(shared_object)
      second_object_config = scope.get_config(shared_object)
      self.assertIn(generic_utils.SHARED_OBJECT_KEY,
                    first_object_config)
      self.assertIn(generic_utils.SHARED_OBJECT_KEY,
                    second_object_config)
      self.assertIs(first_object_config, second_object_config)

  def test_shared_object_loading_scope_noop(self):
    # Test that, without a context manager scope, adding configs will do
    # nothing.
    obj_id = 1
    obj = MaybeSharedObject()
    generic_utils._shared_object_loading_scope().set(obj_id, obj)
    self.assertIsNone(generic_utils._shared_object_loading_scope().get(obj_id))

  def test_shared_object_loading_scope_returns_shared_obj(self):
    obj_id = 1
    obj = MaybeSharedObject()
    with generic_utils.SharedObjectLoadingScope() as scope:
      scope.set(obj_id, obj)
      self.assertIs(scope.get(obj_id), obj)

  def test_nested_shared_object_saving_scopes(self):
    my_obj = MaybeSharedObject()
    with generic_utils.SharedObjectSavingScope() as scope_1:
      scope_1.create_config({}, my_obj)
      with generic_utils.SharedObjectSavingScope() as scope_2:
        # Nesting saving scopes should return the original scope and should
        # not clear any objects we're tracking.
        self.assertIs(scope_1, scope_2)
        self.assertIsNotNone(scope_2.get_config(my_obj))
      self.assertIsNotNone(scope_1.get_config(my_obj))
    self.assertIsNone(generic_utils._shared_object_saving_scope())


if __name__ == '__main__':
  test.main()
