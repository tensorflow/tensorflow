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

from tensorflow.contrib.learn.python.learn.estimators import tensor_signature
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class TensorSignatureTest(test.TestCase):

  def testTensorPlaceholderNone(self):
    self.assertEqual(None,
                     tensor_signature.create_placeholders_from_signatures(None))

  def testTensorSignatureNone(self):
    self.assertEqual(None, tensor_signature.create_signatures(None))

  def testTensorSignatureCompatible(self):
    placeholder_a = array_ops.placeholder(
        name='test', shape=[None, 100], dtype=dtypes.int32)
    placeholder_b = array_ops.placeholder(
        name='another', shape=[256, 100], dtype=dtypes.int32)
    placeholder_c = array_ops.placeholder(
        name='mismatch', shape=[256, 100], dtype=dtypes.float32)
    placeholder_d = array_ops.placeholder(
        name='mismatch', shape=[128, 100], dtype=dtypes.int32)
    signatures = tensor_signature.create_signatures(placeholder_a)
    self.assertTrue(tensor_signature.tensors_compatible(None, None))
    self.assertFalse(tensor_signature.tensors_compatible(None, signatures))
    self.assertFalse(tensor_signature.tensors_compatible(placeholder_a, None))
    self.assertTrue(
        tensor_signature.tensors_compatible(placeholder_a, signatures))
    self.assertTrue(
        tensor_signature.tensors_compatible(placeholder_b, signatures))
    self.assertFalse(
        tensor_signature.tensors_compatible(placeholder_c, signatures))
    self.assertTrue(
        tensor_signature.tensors_compatible(placeholder_d, signatures))

    inputs = {'a': placeholder_a}
    signatures = tensor_signature.create_signatures(inputs)
    self.assertTrue(tensor_signature.tensors_compatible(inputs, signatures))
    self.assertFalse(
        tensor_signature.tensors_compatible(placeholder_a, signatures))
    self.assertFalse(
        tensor_signature.tensors_compatible(placeholder_b, signatures))
    self.assertFalse(
        tensor_signature.tensors_compatible({
            'b': placeholder_b
        }, signatures))
    self.assertTrue(
        tensor_signature.tensors_compatible({
            'a': placeholder_b,
            'c': placeholder_c
        }, signatures))
    self.assertFalse(
        tensor_signature.tensors_compatible({
            'a': placeholder_c
        }, signatures))

  def testSparseTensorCompatible(self):
    t = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    signatures = tensor_signature.create_signatures(t)
    self.assertTrue(tensor_signature.tensors_compatible(t, signatures))

  def testTensorSignaturePlaceholders(self):
    placeholder_a = array_ops.placeholder(
        name='test', shape=[None, 100], dtype=dtypes.int32)
    signatures = tensor_signature.create_signatures(placeholder_a)
    placeholder_out = tensor_signature.create_placeholders_from_signatures(
        signatures)
    self.assertEqual(placeholder_out.dtype, placeholder_a.dtype)
    self.assertTrue(placeholder_out.get_shape().is_compatible_with(
        placeholder_a.get_shape()))
    self.assertTrue(
        tensor_signature.tensors_compatible(placeholder_out, signatures))

    inputs = {'a': placeholder_a}
    signatures = tensor_signature.create_signatures(inputs)
    placeholders_out = tensor_signature.create_placeholders_from_signatures(
        signatures)
    self.assertEqual(placeholders_out['a'].dtype, placeholder_a.dtype)
    self.assertTrue(placeholders_out['a'].get_shape().is_compatible_with(
        placeholder_a.get_shape()))
    self.assertTrue(
        tensor_signature.tensors_compatible(placeholders_out, signatures))

  def testSparseTensorSignaturePlaceholders(self):
    tensor = sparse_tensor.SparseTensor(
        values=[1.0, 2.0], indices=[[0, 2], [0, 3]], dense_shape=[5, 5])
    signature = tensor_signature.create_signatures(tensor)
    placeholder = tensor_signature.create_placeholders_from_signatures(
        signature)
    self.assertTrue(isinstance(placeholder, sparse_tensor.SparseTensor))
    self.assertEqual(placeholder.values.dtype, tensor.values.dtype)

  def testTensorSignatureExampleParserSingle(self):
    examples = array_ops.placeholder(
        name='example', shape=[None], dtype=dtypes.string)
    placeholder_a = array_ops.placeholder(
        name='test', shape=[None, 100], dtype=dtypes.int32)
    signatures = tensor_signature.create_signatures(placeholder_a)
    result = tensor_signature.create_example_parser_from_signatures(signatures,
                                                                    examples)
    self.assertTrue(tensor_signature.tensors_compatible(result, signatures))
    new_signatures = tensor_signature.create_signatures(result)
    self.assertTrue(new_signatures.is_compatible_with(signatures))

  def testTensorSignatureExampleParserDict(self):
    examples = array_ops.placeholder(
        name='example', shape=[None], dtype=dtypes.string)
    placeholder_a = array_ops.placeholder(
        name='test', shape=[None, 100], dtype=dtypes.int32)
    placeholder_b = array_ops.placeholder(
        name='bb', shape=[None, 100], dtype=dtypes.float64)
    inputs = {'a': placeholder_a, 'b': placeholder_b}
    signatures = tensor_signature.create_signatures(inputs)
    result = tensor_signature.create_example_parser_from_signatures(signatures,
                                                                    examples)
    self.assertTrue(tensor_signature.tensors_compatible(result, signatures))
    new_signatures = tensor_signature.create_signatures(result)
    self.assertTrue(new_signatures['a'].is_compatible_with(signatures['a']))
    self.assertTrue(new_signatures['b'].is_compatible_with(signatures['b']))

  def testUnknownShape(self):
    placeholder_unk = array_ops.placeholder(
        name='unk', shape=None, dtype=dtypes.string)
    placeholder_a = array_ops.placeholder(
        name='a', shape=[None], dtype=dtypes.string)
    placeholder_b = array_ops.placeholder(
        name='b', shape=[128, 2], dtype=dtypes.string)
    placeholder_c = array_ops.placeholder(
        name='c', shape=[128, 2], dtype=dtypes.int32)
    unk_signature = tensor_signature.create_signatures(placeholder_unk)
    # Tensors of same dtype match unk shape signature.
    self.assertTrue(
        tensor_signature.tensors_compatible(placeholder_unk, unk_signature))
    self.assertTrue(
        tensor_signature.tensors_compatible(placeholder_a, unk_signature))
    self.assertTrue(
        tensor_signature.tensors_compatible(placeholder_b, unk_signature))
    self.assertFalse(
        tensor_signature.tensors_compatible(placeholder_c, unk_signature))

    string_signature = tensor_signature.create_signatures(placeholder_a)
    int_signature = tensor_signature.create_signatures(placeholder_c)
    # Unk shape Tensor matche signatures same dtype.
    self.assertTrue(
        tensor_signature.tensors_compatible(placeholder_unk, string_signature))
    self.assertFalse(
        tensor_signature.tensors_compatible(placeholder_unk, int_signature))


if __name__ == '__main__':
  test.main()
