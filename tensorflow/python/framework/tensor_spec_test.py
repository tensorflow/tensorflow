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
"""Tests for tensor_spec."""

import pickle

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.core.function import trace_type
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework.type_utils import fulltypes_for_flat_tensors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class TensorSpecTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testDefaultDType(self):
    desc = tensor_spec.TensorSpec([1])
    self.assertEqual(desc.dtype, dtypes.float32)

  def testAcceptsNumpyDType(self):
    desc = tensor_spec.TensorSpec([1], np.float32)
    self.assertEqual(desc.dtype, dtypes.float32)

  def testAcceptsTensorShape(self):
    desc = tensor_spec.TensorSpec(tensor_shape.TensorShape([1]), dtypes.float32)
    self.assertEqual(desc.shape, tensor_shape.TensorShape([1]))

  def testUnknownShape(self):
    desc = tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32)
    self.assertEqual(desc.shape, tensor_shape.TensorShape(None))

  def testShapeCompatibility(self):
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      unknown = array_ops.placeholder(dtypes.int64)
      partial = array_ops.placeholder(dtypes.int64, shape=[None, 1])
      full = array_ops.placeholder(dtypes.int64, shape=[2, 3])
      rank3 = array_ops.placeholder(dtypes.int64, shape=[4, 5, 6])

      desc_unknown = tensor_spec.TensorSpec(None, dtypes.int64)
      self.assertTrue(desc_unknown.is_compatible_with(unknown))
      self.assertTrue(desc_unknown.is_compatible_with(partial))
      self.assertTrue(desc_unknown.is_compatible_with(full))
      self.assertTrue(desc_unknown.is_compatible_with(rank3))

      desc_partial = tensor_spec.TensorSpec([2, None], dtypes.int64)
      self.assertTrue(desc_partial.is_compatible_with(unknown))
      self.assertTrue(desc_partial.is_compatible_with(partial))
      self.assertTrue(desc_partial.is_compatible_with(full))
      self.assertFalse(desc_partial.is_compatible_with(rank3))

      desc_full = tensor_spec.TensorSpec([2, 3], dtypes.int64)
      self.assertTrue(desc_full.is_compatible_with(unknown))
      self.assertFalse(desc_full.is_compatible_with(partial))
      self.assertTrue(desc_full.is_compatible_with(full))
      self.assertFalse(desc_full.is_compatible_with(rank3))

      desc_rank3 = tensor_spec.TensorSpec([4, 5, 6], dtypes.int64)
      self.assertTrue(desc_rank3.is_compatible_with(unknown))
      self.assertFalse(desc_rank3.is_compatible_with(partial))
      self.assertFalse(desc_rank3.is_compatible_with(full))
      self.assertTrue(desc_rank3.is_compatible_with(rank3))

  def testTypeCompatibility(self):
    floats = constant_op.constant(1, dtype=dtypes.float32, shape=[10, 10])
    ints = constant_op.constant(1, dtype=dtypes.int32, shape=[10, 10])
    desc = tensor_spec.TensorSpec(shape=(10, 10), dtype=dtypes.float32)
    self.assertTrue(desc.is_compatible_with(floats))
    self.assertFalse(desc.is_compatible_with(ints))

  def testName(self):
    # Note: "_" isn't a valid tensor name, but it is a valid python symbol
    # name; and tf.function constructs TensorSpecs using function argument
    # names.
    for name in ["beep", "foo/bar:0", "a-b_c/d", "_"]:
      desc = tensor_spec.TensorSpec([1], dtypes.float32, name=name)
      self.assertEqual(desc.name, name)

  def testRepr(self):
    desc1 = tensor_spec.TensorSpec([1], dtypes.float32, name="beep")
    self.assertEqual(
        repr(desc1),
        "TensorSpec(shape=(1,), dtype=tf.float32, name='beep')")
    desc2 = tensor_spec.TensorSpec([1, None], dtypes.int32)
    if desc2.shape._v2_behavior:
      self.assertEqual(
          repr(desc2),
          "TensorSpec(shape=(1, None), dtype=tf.int32, name=None)")
    else:
      self.assertEqual(
          repr(desc2),
          "TensorSpec(shape=(1, ?), dtype=tf.int32, name=None)")

  def testFromTensorSpec(self):
    spec_1 = tensor_spec.TensorSpec((1, 2), dtypes.int32)
    spec_2 = tensor_spec.TensorSpec.from_spec(spec_1)
    self.assertEqual(spec_1, spec_2)

  def testFromTensor(self):
    zero = constant_op.constant(0)
    spec = tensor_spec.TensorSpec.from_tensor(zero)
    self.assertEqual(spec.dtype, dtypes.int32)
    self.assertEqual(spec.shape, [])
    # Tensor.name is meaningless when eager execution is enabled.
    if not context.executing_eagerly():
      self.assertEqual(spec.name, "Const")

  def testFromPlaceholder(self):
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      unknown = array_ops.placeholder(dtypes.int64, name="unknown")
      partial = array_ops.placeholder(dtypes.float32,
                                      shape=[None, 1],
                                      name="partial")

      spec_1 = tensor_spec.TensorSpec.from_tensor(unknown)
      self.assertEqual(spec_1.dtype, dtypes.int64)
      self.assertEqual(spec_1.shape, None)
      self.assertEqual(spec_1.name, "unknown")
      spec_2 = tensor_spec.TensorSpec.from_tensor(partial)
      self.assertEqual(spec_2.dtype, dtypes.float32)
      self.assertEqual(spec_2.shape.as_list(), [None, 1])
      self.assertEqual(spec_2.name, "partial")

  def testFromBoundedTensorSpec(self):
    bounded_spec = tensor_spec.BoundedTensorSpec((1, 2), dtypes.int32, 0, 1)
    spec = tensor_spec.TensorSpec.from_spec(bounded_spec)
    self.assertEqual(bounded_spec.shape, spec.shape)
    self.assertEqual(bounded_spec.dtype, spec.dtype)
    self.assertEqual(bounded_spec.name, spec.name)

  def testSerialization(self):
    desc = tensor_spec.TensorSpec([1, 5], dtypes.float32, "test")
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)

  @test_util.deprecated_graph_mode_only
  def testTypeSpecFromValue(self):
    g = ops.Graph()
    with g.as_default():
      v1 = np.array([1, 2, 3], np.int32)
      t1 = constant_op.constant(v1)

      ops_before = g.get_operations()

      expected = tensor_spec.TensorSpec([3], dtypes.int32)
      self.assertEqual(expected, type_spec.type_spec_from_value(v1))
      self.assertEqual(expected, type_spec.type_spec_from_value(t1))

      # Check that creating TypeSpecs did not require building new Tensors.
      self.assertLen(g.get_operations(), len(ops_before))

  def testEqualTypes(self):
    signature_context = trace_type.InternalTracingContext()
    type_1 = tensor_spec.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    type_2 = tensor_spec.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    self.assertEqual(type_1, type_1)
    self.assertEqual(type_1, type_2)
    self.assertTrue(type_1.is_subtype_of(type_1))
    self.assertTrue(type_2.is_subtype_of(type_1))
    self.assertTrue(type_1.is_subtype_of(type_2))

  def testDtypeMismatch(self):
    signature_context = trace_type.InternalTracingContext()
    type_1 = tensor_spec.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    type_2 = tensor_spec.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.int32,
        None).__tf_tracing_type__(signature_context)
    self.assertNotEqual(type_1, type_2)
    self.assertFalse(type_2.is_subtype_of(type_1))
    self.assertFalse(type_1.is_subtype_of(type_2))

  def testSubtypeOfShapeless(self):
    signature_context = trace_type.InternalTracingContext()
    type_1 = tensor_spec.TensorSpec(
        tensor_shape.TensorShape(None), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    type_2 = tensor_spec.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    self.assertNotEqual(type_1, type_2)
    self.assertFalse(type_1.is_subtype_of(type_2))
    self.assertTrue(type_2.is_subtype_of(type_1))

  def testSubtypeOfDimlessShape(self):
    signature_context = trace_type.InternalTracingContext()
    type_1 = tensor_spec.TensorSpec(
        tensor_shape.TensorShape([None, None, None]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    type_2 = tensor_spec.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    self.assertNotEqual(type_1, type_2)
    self.assertFalse(type_1.is_subtype_of(type_2))
    self.assertTrue(type_2.is_subtype_of(type_1))

  def testFlatTensorSpecs(self):
    spec = tensor_spec.TensorSpec([1], np.float32)
    self.assertEqual(spec._flat_tensor_specs, [spec])

  def testFullTypesForFlatTensors(self):
    spec = tensor_spec.TensorSpec([1], np.float32)
    full_type_list = fulltypes_for_flat_tensors(spec)
    expect = [full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET)]
    self.assertEqual(len(spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)


class BoundedTensorSpecTest(test_util.TensorFlowTestCase):

  def testInvalidMinimum(self):
    with self.assertRaisesRegex(ValueError, "not compatible"):
      tensor_spec.BoundedTensorSpec((3, 5), dtypes.uint8, (0, 0, 0), (1, 1))

  def testInvalidMaximum(self):
    with self.assertRaisesRegex(ValueError, "not compatible"):
      tensor_spec.BoundedTensorSpec((3, 5), dtypes.uint8, 0, (1, 1, 1))

  def testMinimumMaximumAttributes(self):
    spec = tensor_spec.BoundedTensorSpec(
        (1, 2, 3), dtypes.float32, 0, (5, 5, 5))
    self.assertEqual(type(spec.minimum), np.ndarray)
    self.assertEqual(type(spec.maximum), np.ndarray)
    self.assertAllEqual(spec.minimum, np.array(0, dtype=np.float32))
    self.assertAllEqual(spec.maximum, np.array([5, 5, 5], dtype=np.float32))

  def testNotWriteableNP(self):
    spec = tensor_spec.BoundedTensorSpec(
        (1, 2, 3), dtypes.float32, 0, (5, 5, 5))
    with self.assertRaisesRegex(ValueError, "read-only"):
      spec.minimum[0] = -1
    with self.assertRaisesRegex(ValueError, "read-only"):
      spec.maximum[0] = 100

  def testReuseSpec(self):
    spec_1 = tensor_spec.BoundedTensorSpec((1, 2), dtypes.int32,
                                           minimum=0, maximum=1)
    spec_2 = tensor_spec.BoundedTensorSpec(
        spec_1.shape, spec_1.dtype, spec_1.minimum, spec_1.maximum)
    self.assertEqual(spec_1, spec_2)

  def testScalarBounds(self):
    spec = tensor_spec.BoundedTensorSpec(
        (), dtypes.float32, minimum=0.0, maximum=1.0)

    self.assertIsInstance(spec.minimum, np.ndarray)
    self.assertIsInstance(spec.maximum, np.ndarray)

    # Sanity check that numpy compares correctly to a scalar for an empty shape.
    self.assertEqual(0.0, spec.minimum)
    self.assertEqual(1.0, spec.maximum)

    # Check that the spec doesn't fail its own input validation.
    _ = tensor_spec.BoundedTensorSpec(
        spec.shape, spec.dtype, spec.minimum, spec.maximum)

  def testFromBoundedTensorSpec(self):
    spec_1 = tensor_spec.BoundedTensorSpec((1, 2), dtypes.int32,
                                           minimum=0, maximum=1)
    spec_2 = tensor_spec.BoundedTensorSpec.from_spec(spec_1)
    self.assertEqual(spec_1, spec_2)

  def testEquality(self):
    spec_1_1 = tensor_spec.BoundedTensorSpec((1, 2, 3), dtypes.float32,
                                             0, (5, 5, 5))
    spec_1_2 = tensor_spec.BoundedTensorSpec((1, 2, 3), dtypes.float32,
                                             0.00000001,
                                             (5, 5, 5.00000000000000001))
    spec_2_1 = tensor_spec.BoundedTensorSpec((1, 2, 3), dtypes.float32,
                                             1, (5, 5, 5))
    spec_2_2 = tensor_spec.BoundedTensorSpec((1, 2, 3), dtypes.float32,
                                             (1, 1, 1), (5, 5, 5))
    spec_2_3 = tensor_spec.BoundedTensorSpec((1, 2, 3), dtypes.float32,
                                             (1, 1, 1), 5)
    spec_3_1 = tensor_spec.BoundedTensorSpec((1, 2, 3), dtypes.float32,
                                             (2, 1, 1), (5, 5, 5))

    self.assertEqual(spec_1_1, spec_1_2)
    self.assertEqual(spec_1_2, spec_1_1)

    self.assertNotEqual(spec_1_1, spec_2_2)
    self.assertNotEqual(spec_1_1, spec_2_1)
    self.assertNotEqual(spec_2_2, spec_1_1)
    self.assertNotEqual(spec_2_1, spec_1_1)

    self.assertEqual(spec_2_1, spec_2_2)
    self.assertEqual(spec_2_2, spec_2_1)
    self.assertEqual(spec_2_2, spec_2_3)

    self.assertNotEqual(spec_1_1, spec_3_1)
    self.assertNotEqual(spec_2_1, spec_3_1)
    self.assertNotEqual(spec_2_2, spec_3_1)

  def testFromTensorSpec(self):
    spec = tensor_spec.TensorSpec((1, 2), dtypes.int32)
    bounded_spec = tensor_spec.BoundedTensorSpec.from_spec(spec)
    self.assertEqual(spec.shape, bounded_spec.shape)
    self.assertEqual(spec.dtype, bounded_spec.dtype)
    self.assertEqual(spec.dtype.min, bounded_spec.minimum)
    self.assertEqual(spec.dtype.max, bounded_spec.maximum)
    self.assertEqual(spec.name, bounded_spec.name)

  def testSerialization(self):
    desc = tensor_spec.BoundedTensorSpec([1, 5], dtypes.float32, -1, 1, "test")
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)


if __name__ == "__main__":
  googletest.main()
