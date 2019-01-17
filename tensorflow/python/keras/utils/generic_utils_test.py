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
"""Tests for Keras generic Python utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras
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


class SliceArraysTest(test.TestCase):

  def test_slice_arrays(self):
    input_a = list([1, 2, 3])
    self.assertEqual(keras.utils.generic_utils.slice_arrays(input_a, start=0),
                     [None, None, None])
    self.assertEqual(keras.utils.generic_utils.slice_arrays(input_a, stop=3),
                     [None, None, None])
    self.assertEqual(keras.utils.generic_utils.slice_arrays(
        input_a, start=0, stop=1),
                     [None, None, None])


if __name__ == '__main__':
  test.main()
