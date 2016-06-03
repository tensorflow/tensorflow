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

"""Tests for learn.estimators.tensor_signature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


from tensorflow.contrib.learn.python.learn.estimators import tensor_signature


class TensorSignatureTest(tf.test.TestCase):

  def testTensorSignatureCompatible(self):
    placeholder_a = tf.placeholder(name='test',
                                   shape=[None, 100],
                                   dtype=tf.int32)
    placeholder_b = tf.placeholder(name='another',
                                   shape=[256, 100],
                                   dtype=tf.int32)
    placeholder_c = tf.placeholder(name='mismatch',
                                   shape=[256, 100],
                                   dtype=tf.float32)
    placeholder_d = tf.placeholder(name='mismatch',
                                   shape=[128, 100],
                                   dtype=tf.int32)
    signatures = tensor_signature.create_signatures(placeholder_a)
    self.assertTrue(tensor_signature.tensors_compatible(placeholder_a,
                                                        signatures))
    self.assertTrue(tensor_signature.tensors_compatible(placeholder_b,
                                                        signatures))
    self.assertFalse(tensor_signature.tensors_compatible(placeholder_c,
                                                         signatures))
    self.assertTrue(tensor_signature.tensors_compatible(placeholder_d,
                                                        signatures))

    inputs = {'a': placeholder_a}
    signatures = tensor_signature.create_signatures(inputs)
    self.assertTrue(tensor_signature.tensors_compatible(inputs, signatures))
    self.assertFalse(tensor_signature.tensors_compatible(placeholder_a,
                                                         signatures))
    self.assertFalse(tensor_signature.tensors_compatible(placeholder_b,
                                                         signatures))
    self.assertFalse(tensor_signature.tensors_compatible(
        {'b': placeholder_b}, signatures))
    self.assertTrue(tensor_signature.tensors_compatible(
        {'a': placeholder_b,
         'c': placeholder_c}, signatures))
    self.assertFalse(tensor_signature.tensors_compatible(
        {'a': placeholder_c}, signatures))

  def testSparseTensorCompatible(self):
    t = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], shape=[3, 4])
    signatures = tensor_signature.create_signatures(t)
    self.assertTrue(tensor_signature.tensors_compatible(t, signatures))

  def testTensorSignaturePlaceholders(self):
    placeholder_a = tf.placeholder(name='test',
                                   shape=[None, 100],
                                   dtype=tf.int32)
    signatures = tensor_signature.create_signatures(placeholder_a)
    placeholder_out = tensor_signature.create_placeholders_from_signatures(
        signatures)
    self.assertEqual(placeholder_out.dtype, placeholder_a.dtype)
    self.assertTrue(placeholder_out.get_shape().is_compatible_with(
        placeholder_a.get_shape()))
    self.assertTrue(tensor_signature.tensors_compatible(placeholder_out,
                                                        signatures))

    inputs = {'a': placeholder_a}
    signatures = tensor_signature.create_signatures(inputs)
    placeholders_out = tensor_signature.create_placeholders_from_signatures(
        signatures)
    self.assertEqual(placeholders_out['a'].dtype, placeholder_a.dtype)
    self.assertTrue(
        placeholders_out['a'].get_shape().is_compatible_with(
            placeholder_a.get_shape()))
    self.assertTrue(tensor_signature.tensors_compatible(placeholders_out,
                                                        signatures))

  def testSparseTensorSignaturePlaceholders(self):
    tensor = tf.SparseTensor(values=[1.0, 2.0], indices=[[0, 2], [0, 3]],
                             shape=[5, 5])
    signature = tensor_signature.create_signatures(tensor)
    placeholder = tensor_signature.create_placeholders_from_signatures(
        signature)
    self.assertTrue(isinstance(placeholder, tf.SparseTensor))
    self.assertEqual(placeholder.values.dtype, tensor.values.dtype)


if __name__ == '__main__':
  tf.test.main()
