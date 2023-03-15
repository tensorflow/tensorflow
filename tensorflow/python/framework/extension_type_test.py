# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.framework.extension_type."""

import contextlib
import copy
import pickle
import tempfile
import typing

from absl.testing import parameterized
import typing_extensions

from tensorflow.core.framework import full_type_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.framework.type_utils import fulltypes_for_flat_tensors
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect


POSITIONAL_OR_KEYWORD = tf_inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD_ONLY = tf_inspect.Parameter.KEYWORD_ONLY


class MaskedTensorV1(extension_type.ExtensionType):
  """Example subclass of ExtensionType, used for testing."""
  values: ops.Tensor
  mask: ops.Tensor


class MaskedTensorV2(extension_type.ExtensionType):
  """Example subclass of ExtensionType, used for testing.

  This version adds methods, classmethod, staticmethod, and properties, and
  customizes `__repr__` and `__validate__`.  It also adds a `__name__` field,
  which enables serialization.
  """
  __name__ = 'tf.test.MaskedTensorV2'

  values: ops.Tensor
  mask: ops.Tensor

  def __repr__(self):
    if hasattr(self.values, 'numpy') and hasattr(self.mask, 'numpy'):
      return '<MaskedTensorV2 %s>' % _masked_array_repr(self.values.numpy(),
                                                        self.mask.numpy())
    else:
      return super(MaskedTensorV2, self).__repr__()

  @property
  def shape(self):
    return self.values.shape

  @property
  def dtype(self):
    return self.values.dtype

  @classmethod
  def from_full_tensor(cls, values):
    return cls(values, array_ops.ones_like(values, dtype=dtypes.bool))

  # A dummy example to test support of staticmethod
  @staticmethod
  def doc_link():
    return 'http://example.com/masked_tensor'

  def __validate__(self):
    self.values.shape.assert_is_compatible_with(self.mask.shape)

  def with_default(self, default):
    return array_ops.where_v2(self.mask, self.values, default)

  __add__ = math_ops.add
  __sub__ = math_ops.subtract


def _masked_array_repr(values, mask):
  """Returns a string representation for a masked numpy array."""
  assert len(values) == len(mask)
  if len(values.shape) == 1:
    items = [repr(v) if m else '_' for (v, m) in zip(values, mask)]
  else:
    items = [_masked_array_repr(v, m) for (v, m) in zip(values, mask)]
  return '[%s]' % ', '.join(items)


class MaskedTensorV3(extension_type.BatchableExtensionType):
  """Example subclass of ExtensionType, used for testing.

  This version adds Keras required properties to MaskedTensor and its Spec
  class, to test Keras integration.
  """
  __name__ = 'tf.test.MaskedTensorV3.Spec'

  values: typing.Union[ops.Tensor, ragged_tensor.RaggedTensor]
  mask: typing.Union[ops.Tensor, ragged_tensor.RaggedTensor]

  def __init__(self, values, mask):
    if isinstance(values, ragged_tensor.RaggedTensor):
      assert isinstance(mask, ragged_tensor.RaggedTensor)
      assert mask.dtype == dtypes.bool
    else:
      values = ops.convert_to_tensor(values)
      mask = ops.convert_to_tensor(mask, dtypes.bool)
    self.values = values
    self.mask = mask

  # Required by assert_input_compatibility in keras/engine/input_spec.py
  @property
  def shape(self):
    return self.values.shape

  @property
  def dtype(self):
    return self.values.dtype

  class Spec:

    # Required by KerasTensor.shape in keras/engine/keras_tensor.py
    @property
    def _shape(self):
      return self.values._shape


class ForwardRefA(extension_type.ExtensionType):
  x: typing.Tuple[typing.Union['ForwardRefA', 'ForwardRefB'], ...]
  y: 'ForwardRefB'


class ForwardRefB(extension_type.ExtensionType):
  z: 'ForwardRefB'
  n: ops.Tensor


class ExtensionTypeWithTensorDefault(extension_type.ExtensionType):
  x: ops.Tensor = 5
  y: ops.Tensor = ['a', 'b', 'c']


@test_util.run_all_in_graph_and_eager_modes
class ExtensionTypeTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testAttributeAccessors(self):
    mt1 = MaskedTensorV2([1, 2, 3, 4], [True, True, False, True])
    mt2 = extension_type.pack(mt1)

    for mt in [mt1, mt2]:
      self.assertIsInstance(mt.values, ops.Tensor)
      self.assertAllEqual(mt.values, [1, 2, 3, 4])
      self.assertIsInstance(mt.mask, ops.Tensor)
      self.assertAllEqual(mt.mask, [True, True, False, True])

  def testAttributesAreImmutable(self):
    mt1 = MaskedTensorV2([1, 2, 3, 4], [True, True, False, True])
    mt2 = extension_type.pack(mt1)

    for mt in [mt1, mt2]:
      with self.assertRaisesRegex(
          AttributeError,
          'Cannot mutate attribute `score` outside the custom constructor of ExtensionType'
      ):
        mt.score = 12
      with self.assertRaisesRegex(
          AttributeError,
          'Cannot mutate attribute `values` outside the custom constructor of ExtensionType'
      ):
        mt.values = constant_op.constant([4, 3, 2, 1])
      with self.assertRaisesRegex(
          AttributeError,
          'Cannot mutate attribute `values` outside the custom constructor of ExtensionType'
      ):
        del mt.values

  def testClassAndStaticMethod(self):
    mt = MaskedTensorV2.from_full_tensor([1, 2, 3, 4])
    self.assertAllEqual(mt.mask, [True, True, True, True])
    self.assertEqual(mt.doc_link(), 'http://example.com/masked_tensor')

  def testRepr(self):
    values = constant_op.constant([1, 2, 3, 4])
    mask = constant_op.constant([True, True, False, True])
    mt = MaskedTensorV1(values, mask)
    expected = f'MaskedTensorV1(values={values!r}, mask={mask!r})'
    self.assertEqual(expected, repr(mt))

  def testEagerRepr(self):
    values = constant_op.constant([1, 2, 3, 4])
    mask = constant_op.constant([True, True, False, True])
    mt = MaskedTensorV2(values, mask)
    if context.executing_eagerly():
      expected = '<MaskedTensorV2 [1, 2, _, 4]>'
    else:
      expected = f'MaskedTensorV2(values={values!r}, mask={mask!r})'

    self.assertEqual(expected, repr(mt))
    self.assertEqual(expected, repr(mt))

  def testAsDict(self):
    values = constant_op.constant([1, 2, 3, 4])
    mask = constant_op.constant([True, True, False, True])
    mt = MaskedTensorV1(values, mask)
    mt_dict = extension_type.as_dict(mt)
    self.assertEqual({'values': values, 'mask': mask}, mt_dict)

  def testConstructorSignature(self):

    class MyType(extension_type.ExtensionType):
      x: ops.Tensor
      y: ops.Tensor
      z: typing.Tuple[typing.Union[int, str], ...] = [1, 'two', 3]

    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('x', POSITIONAL_OR_KEYWORD, annotation=ops.Tensor),
        tf_inspect.Parameter('y', POSITIONAL_OR_KEYWORD, annotation=ops.Tensor),
        tf_inspect.Parameter(
            'z',
            POSITIONAL_OR_KEYWORD,
            annotation=typing.Tuple[typing.Union[int, str], ...],
            default=(1, 'two', 3)),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyType)
    self.assertEqual(expected_sig, tf_inspect.signature(MyType.__init__))

  def testConstructorSignatureWithKeywordOnlyArgs(self):

    class MyType(extension_type.ExtensionType):
      a: int
      b: str = 'Hello world'
      c: ops.Tensor

    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('a', POSITIONAL_OR_KEYWORD, annotation=int),
        tf_inspect.Parameter(
            'b', POSITIONAL_OR_KEYWORD, annotation=str, default='Hello world'),
        tf_inspect.Parameter('c', KEYWORD_ONLY, annotation=ops.Tensor),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyType)
    self.assertEqual(expected_sig, tf_inspect.signature(MyType.__init__))

  def testConstructorSignatureWithDefaultForTensorField(self):
    a = ExtensionTypeWithTensorDefault()

    # Check that the default values were *not* converted to Tensors:
    sig = tf_inspect.signature(ExtensionTypeWithTensorDefault.__init__)
    self.assertIsInstance(sig.parameters['x'].default, int)
    self.assertIsInstance(sig.parameters['y'].default, list)

    # The following would fail with "RuntimeError: Attempting to capture an
    # EagerTensor without building a function" if we converted the default
    # value to a Tensor when we built the type.
    self.assertAllEqual(a.x + constant_op.constant(3), 8)

  def testConstructorSignatureWithAnnotatedTensorField(self):

    class MyType(extension_type.ExtensionType):
      a: typing_extensions.Annotated[ops.Tensor, 'metadata']
      b: typing_extensions.Annotated[str, 'metadata'] = 'Hello world'
      c: typing.Optional[typing_extensions.Annotated[int, 'metadata']] = None

    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('a', POSITIONAL_OR_KEYWORD, annotation=ops.Tensor),
        tf_inspect.Parameter(
            'b', POSITIONAL_OR_KEYWORD, annotation=str, default='Hello world'),
        tf_inspect.Parameter(
            'c',
            POSITIONAL_OR_KEYWORD,
            annotation=typing.Optional[int],
            default=None),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyType)
    self.assertEqual(expected_sig, tf_inspect.signature(MyType.__init__))

  def testEmptyType(self):

    class EmptyType(extension_type.ExtensionType):
      pass

    self.assertEmpty(EmptyType._tf_extension_type_fields())
    x = EmptyType()
    self.assertEqual(
        repr(x), 'ExtensionTypeTest.testEmptyType.<locals>.EmptyType()')

  def testCustomConstrutor(self):

    class SummarizedTensor(extension_type.ExtensionType):
      values: ops.Tensor
      mean: ops.Tensor
      max: ops.Tensor

      def __init__(self, values):
        self.values = ops.convert_to_tensor(values)
        self.mean = math_ops.reduce_mean(values)
        self.max = math_ops.reduce_max(values)

    x = SummarizedTensor([[1.0, 2, 3], [4, 5, 6]])
    self.assertAllEqual(x.values, [[1.0, 2, 3], [4, 5, 6]])
    self.assertAllEqual(x.mean, 3.5)
    self.assertAllEqual(x.max, 6)

  class Node(extension_type.ExtensionType):
    x: ops.Tensor
    y: typing.Optional[str] = None
    children: typing.Tuple['ExtensionTypeTest.Node', ...] = ()

  def testConstructorWithDefaultValues(self):
    a = ExtensionTypeTest.Node(5)
    self.assertAllEqual(a.x, 5)
    self.assertIsNone(a.y)
    self.assertEqual(a.children, ())

    b = ExtensionTypeTest.Node(6, 'blue')
    self.assertAllEqual(b.x, 6)
    self.assertEqual(b.y, 'blue')
    self.assertEqual(b.children, ())

    c = ExtensionTypeTest.Node(7, children=(a, b))
    self.assertAllEqual(c.x, 7)
    self.assertIsNone(c.y)
    self.assertEqual(c.children, (a, b))

  def testCustomConstrutorCantMutateNestedValues(self):

    class Foo(extension_type.ExtensionType):
      x: int

    class Bar(extension_type.ExtensionType):
      foo: Foo

      def __init__(self, foo):
        foo.x = 33  # This raises an exception

    with self.assertRaisesRegex(
        AttributeError,
        'Cannot mutate attribute `x` outside the custom constructor of ExtensionType'
    ):
      Bar(Foo(12))

  def testCustomValidate(self):

    class AlignedTensors(extension_type.ExtensionType):
      x: ops.Tensor
      y: ops.Tensor

      def __validate__(self):
        self.x.shape.assert_is_compatible_with(self.y.shape)

    aligned = AlignedTensors([1, 2, 3], ['a', 'b', 'c'])
    self.assertAllEqual(aligned.x, [1, 2, 3])
    self.assertAllEqual(aligned.y, [b'a', b'b', b'c'])

    with self.assertRaises(ValueError):
      AlignedTensors([1, 2, 3], ['a', 'b', 'c', 'd'])

  def testEquals(self):

    class MyType(extension_type.ExtensionType):
      values: ops.Tensor
      score: ops.Tensor
      flavor: str

    x1 = MyType([1, 2], 8, 'blue')
    x2 = MyType([1, 2], 8, 'blue')
    y = MyType([1, 2], 8, 'red')
    z = MyType([1, 2], 7, 'blue')
    self.assertAllEqual(x1 == x2, True)
    self.assertAllEqual(x1 != x2, False)
    self.assertAllEqual(x1 == y, False)
    self.assertAllEqual(x1 != y, True)
    self.assertAllEqual(x1 == z, False)
    self.assertAllEqual(y == z, False)

    # These are not equal, even though their values are broadcast-compatible
    # and elements are all equal when we broadcast.  Shapes must match.
    a = MyType([1, 1, 1, 1], 0, 'x')
    b = MyType([[1, 1, 1, 1]], 0, 'x')
    c = MyType([[1, 1], [1, 1]], 0, 'x')
    self.assertAllEqual(a == b, False)
    self.assertAllEqual(a == c, False)
    self.assertAllEqual(b == c, False)

    # Test with unknown shapes (executes a different codepath).
    a_ph = replace_tensors_with_placeholders(a)
    b_ph = replace_tensors_with_placeholders(b)
    c_ph = replace_tensors_with_placeholders(c)
    self.assertAllEqual(a_ph == b_ph, False)
    self.assertAllEqual(a_ph == c_ph, False)
    self.assertAllEqual(b_ph == c_ph, False)

  def testPassIntoTfFunction(self):

    @def_function.function
    def fn(x):
      return x.with_default(99)

    mt = MaskedTensorV2([1, 2, 3, 4], [True, True, False, True])
    self.assertAllEqual([1, 2, 99, 4], fn(mt))
    self.assertAllEqual([1, 2, 99, 4], fn(extension_type.pack(mt)))

  def testReturnFromTfFunction(self):

    @def_function.function
    def mask_neg_values(x):
      return MaskedTensorV2(x, x > 0)

    @def_function.function
    def mask_neg_values_packed(x):
      return extension_type.pack(MaskedTensorV2(x, x > 0))

    expected = MaskedTensorV2([5, 8, -3, 9], [True, True, False, True])

    actual1 = mask_neg_values(constant_op.constant([5, 8, -3, 9]))
    self.assertIsInstance(actual1, MaskedTensorV2)
    self.assertAllEqual(expected.values, actual1.values)
    self.assertAllEqual(expected.mask, actual1.mask)

    actual2 = mask_neg_values_packed(constant_op.constant([5, 8, -3, 9]))
    self.assertIsInstance(actual2, MaskedTensorV2)
    self.assertTrue(extension_type.is_packed(actual2))
    self.assertAllEqual(expected.values, actual2.values)
    self.assertAllEqual(expected.mask, actual2.mask)

  def testCaptureByTfFunction(self):
    x = MaskedTensorV2(
        values=[[1, 2, 3], [4, 5, 6]],
        mask=[[True, True, True], [True, False, True]])

    @def_function.function
    def add_to_x(y):
      return MaskedTensorV2(x.values + y.values, x.mask & y.mask)

    actual = add_to_x(MaskedTensorV2([10, 20, 30], [False, True, True]))
    expected = MaskedTensorV2(
        values=[[11, 22, 33], [14, 25, 36]],
        mask=[[False, True, True], [False, False, True]])
    self.assertIsInstance(actual, MaskedTensorV2)
    self.assertAllEqual(expected.values, actual.values)
    self.assertAllEqual(expected.mask, actual.mask)

  def testTfFunctionArgMutationError(self):

    @def_function.function
    def fn_with_side_effect(mts):
      mts.append(MaskedTensorV1(mts[0].values * 2, mts[0].mask))

    with self.assertRaisesRegex(ValueError, 'should not modify'):
      fn_with_side_effect([MaskedTensorV1([10, 20, 30], [False, True, True])])

  def testNestPackUnpack(self):

    class CandyStore(extension_type.ExtensionType):
      name: ops.Tensor
      prices: typing.Mapping[str, ops.Tensor]

    store = CandyStore('Yum', {'gum': [0.42, 0.48], 'chocolate': [0.83, 1.02]})
    components = nest.flatten(store, expand_composites=True)
    repacked_1 = nest.pack_sequence_as(
        store, components, expand_composites=True)
    repacked_2 = nest.pack_sequence_as(
        store._type_spec, components, expand_composites=True)

    # Note: dicts get sorted by key.
    self.assertLen(components, 3)
    self.assertAllEqual(components[0], b'Yum')
    self.assertAllClose(components[1], [0.83, 1.02])
    self.assertAllClose(components[2], [0.42, 0.48])

    for repacked in [repacked_1, repacked_2]:
      self.assertAllEqual(repacked.name, b'Yum')
      self.assertAllClose(repacked.prices['gum'], [0.42, 0.48])
      self.assertAllClose(repacked.prices['chocolate'], [0.83, 1.02])

  def testSimpleCond(self):
    x = MaskedTensorV1([1, 2, 3, 4], [True, False, True, False])
    y = MaskedTensorV1([5, 6, 7, 8], [False, True, True, False])

    x_2 = control_flow_ops.cond(
        constant_op.constant(True), lambda: x, lambda: y)
    y_2 = control_flow_ops.cond(
        constant_op.constant(False), lambda: x, lambda: y)

    self.assertAllEqual(x.values, x_2.values)
    self.assertAllEqual(x.mask, x_2.mask)
    self.assertAllEqual(y.values, y_2.values)
    self.assertAllEqual(y.mask, y_2.mask)

  def testComplexCond(self):
    mt = MaskedTensorV1([1, 2, 3, 4], [True, False, True, False])

    def true_fn():
      return MaskedTensorV1(
          array_ops.where_v2(mt.mask, mt.values, -1), mt.values > 3)

    def false_fn():
      return MaskedTensorV1(
          array_ops.where_v2(mt.mask, 100, mt.values * 2),
          math_ops.logical_not(mt.mask))

    x = control_flow_ops.cond(constant_op.constant(True), true_fn, false_fn)
    y = control_flow_ops.cond(constant_op.constant(False), true_fn, false_fn)

    self.assertAllEqual(x.values, [1, -1, 3, -1])
    self.assertAllEqual(x.mask, [False, False, False, True])
    self.assertAllEqual(y.values, [100, 4, 100, 8])
    self.assertAllEqual(y.mask, [False, True, False, True])

  def testCondAutograph(self):

    @def_function.function
    def fn(mt):
      if mt.values[3] > 3:
        return MaskedTensorV1(
            array_ops.where_v2(mt.mask, mt.values, -1), mt.values > 3)
      else:
        return MaskedTensorV1(
            array_ops.where_v2(mt.mask, 100, mt.values * 2), not mt.mask)

    x = fn(MaskedTensorV1([1, 2, 3, 4], [True, False, True, False]))
    self.assertAllEqual(x.values, [1, -1, 3, -1])
    self.assertAllEqual(x.mask, [False, False, False, True])

  def testCondTypeMismatch(self):
    if context.executing_eagerly:
      # In eager mode, tf.cond eagerly runs either true_fn or false_fn, and
      # ignores the other one; so it doesn't detect any type mismatches
      # between the two outcomes.  (See _eager_cond_implementation in
      # control_flow_ops.py.)
      return

    a = lambda: MaskedTensorV1([1, 2, 3], [True, True, False])
    b = lambda: MaskedTensorV1(['a', 'b', 'c'], [False, True, True])
    c = lambda: MaskedTensorV2([4, 5, 6], [True, True, False])
    d = lambda: constant_op.constant([7, 8, 9])

    with self.assertRaisesRegex(
        ValueError,
        'Incompatible return values of true_fn and false_fn: The two '
        "structures don't have the same nested structure"):
      control_flow_ops.cond(constant_op.constant(True), a, b)
    with self.assertRaisesRegex(
        TypeError, 'Incompatible return types of true_fn and false_fn: The two '
        "structures don't have the same nested structure"):
      control_flow_ops.cond(constant_op.constant(True), a, c)
    with self.assertRaisesRegex(
        ValueError,
        'Incompatible return values of true_fn and false_fn: The two '
        "structures don't have the same nested structure"):
      control_flow_ops.cond(constant_op.constant(True), a, d)

  def testCondPacked(self):
    x = MaskedTensorV2([1, 2, 3, 4], [True, False, True, False])
    y = MaskedTensorV2([5, 6, 7, 8], [False, True, True, False])
    x = extension_type.pack(x)
    y = extension_type.pack(y)

    x_2 = control_flow_ops.cond(
        constant_op.constant(True), lambda: x, lambda: y)
    y_2 = control_flow_ops.cond(
        constant_op.constant(False), lambda: x, lambda: y)

    self.assertAllEqual(x.values, x_2.values)
    self.assertAllEqual(x.mask, x_2.mask)
    self.assertAllEqual(y.values, y_2.values)
    self.assertAllEqual(y.mask, y_2.mask)

    a = MaskedTensorV2([1, 2, 3, 4], [True, False, True, False])
    b = extension_type.pack(a)
    b = control_flow_ops.cond(
        constant_op.constant(True), lambda: array_ops.size(a.mask),
        lambda: array_ops.size(a.values))
    self.assertAllEqual(b, 4)

    # Note: the following example would fail (with `Retval[0] does not have a
    # value`) if `ExtensionType.__getattr__` cached the results of unpacking
    # the value.  See the comment in `ExtensionType.__getattr__` for details.
    c = MaskedTensorV2([1, 2, 3, 4], [True, False, True, False])
    c = extension_type.pack(c)
    d = control_flow_ops.cond(
        constant_op.constant(False), lambda: array_ops.size(c.mask),
        lambda: array_ops.size(c.values))
    self.assertAllEqual(d, 4)

  def testWhileLoop(self):
    x = MaskedTensorV1([1, 2, 3, 4], [True, False, True, False])

    cond = lambda i, x: i < 10
    body = lambda i, x: (i + 1, MaskedTensorV1(x.values * 2, x.mask))
    _, y = while_loop.while_loop_v2(cond, body, [0, x])

    self.assertIsInstance(y, MaskedTensorV1)
    self.assertAllEqual(y.values, [1024, 2048, 3072, 4096])
    self.assertAllEqual(y.mask, [True, False, True, False])

  def testWhileLoopAutograph(self):

    @def_function.function
    def fn(x, n):
      for _ in math_ops.range(n):
        x = MaskedTensorV1(x.values * 2, x.mask)
      return x

    y = fn(MaskedTensorV1([1, 2, 3, 4], [True, False, True, False]), 10)
    self.assertIsInstance(y, MaskedTensorV1)
    self.assertAllEqual(y.values, [1024, 2048, 3072, 4096])
    self.assertAllEqual(y.mask, [True, False, True, False])

  def testWhileLoopTypeMismatch(self):
    x = MaskedTensorV1([1, 2, 3, 4], [True, False, True, False])

    cond = lambda i, x: i < 10

    def body(i, x):
      if isinstance(x, MaskedTensorV1):
        return x.values * 2
      else:
        return MaskedTensorV1(x, x > i)

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure"):
      while_loop.while_loop_v2(cond, body, [0, x])

  def testWhileLoopPacked(self):
    x = MaskedTensorV2([1, 2, 3, 4], [True, False, True, False])
    x = extension_type.pack(x)
    cond = lambda i, x: i < 10

    def body(i, x):
      return i + 1, extension_type.pack(MaskedTensorV2(x.values * 2, x.mask))

    _, y = while_loop.while_loop_v2(cond, body, [0, x])
    self.assertIsInstance(y, MaskedTensorV2)
    self.assertAllEqual(y.values, [1024, 2048, 3072, 4096])
    self.assertAllEqual(y.mask, [True, False, True, False])

  def testNestedFields(self):
    PossiblyRaggedTensor = typing.Union[ops.Tensor, ragged_tensor.RaggedTensor]
    ToyFeatures = typing.Mapping[str, PossiblyRaggedTensor]

    class ToyInfo(extension_type.ExtensionType):
      version: str
      toys: typing.Tuple[typing.Tuple[str, ops.Tensor, ToyFeatures], ...]
      boxes: typing.Mapping[str, ops.Tensor]

    authors = [[b'A', b'Aardvark'], [b'Z', b'Zhook']]
    toys = [('car', 1.0, {
        'size': [8, 3, 2],
        'color': [0.3, 0.2, 0.8]
    }), ('book', 3.7, {
        'authors': ragged_factory_ops.constant(authors)
    })]
    boxes = {'green': ['car'], 'blue': ['car', 'book', 'book']}
    toy_info = ToyInfo(version='1.0 alpha', toys=toys, boxes=boxes)

    self.assertEqual(toy_info.version, '1.0 alpha')
    self.assertEqual(toy_info.toys[0][0], 'car')
    self.assertIsInstance(toy_info.toys[0][1], ops.Tensor)
    self.assertAllEqual(toy_info.toys[0][1], 1.0)
    self.assertEqual(set(toy_info.toys[0][2].keys()), {'size', 'color'})
    self.assertIsInstance(toy_info.toys[0][2]['size'], ops.Tensor)
    self.assertAllEqual(toy_info.toys[0][2]['size'], [8, 3, 2])
    self.assertIsInstance(toy_info.toys[1][2]['authors'],
                          ragged_tensor.RaggedTensor)
    self.assertAllEqual(toy_info.toys[1][2]['authors'], authors)
    self.assertAllEqual(toy_info.boxes['green'], [b'car'])
    self.assertAllEqual(toy_info.boxes['blue'], ['car', 'book', 'book'])

    expected_repr = (
        r"ToyInfo\(version='1.0 alpha', toys=\("
        r"\('car', <tf.Tensor[^>]*>, ImmutableDict\("
        r"{'size': <tf.Tensor[^>]*>, 'color': <tf.Tensor[^>]*>}\)\), "
        r"\('book', <tf.Tensor[^>]*>, ImmutableDict\("
        r"{'authors': (<tf.RaggedTensor[^>]*>|tf.RaggedTensor\(.*\))}\)\)\), "
        r'boxes=ImmutableDict\('
        r"{'green': <tf.Tensor[^>]*>, 'blue': <tf.Tensor[^>]*>}\)\)")

    self.assertRegex(repr(toy_info), expected_repr)

  def testNestedExtensionTypes(self):
    PossiblyMaskedTensor = typing.Union[ops.Tensor, MaskedTensorV1]

    class Toy(extension_type.ExtensionType):
      name: str
      price: ops.Tensor
      features: typing.Mapping[str, PossiblyMaskedTensor]

    class Box(extension_type.ExtensionType):
      contents: ops.Tensor

    class ToyInfo(extension_type.ExtensionType):
      version: str
      toys: typing.Tuple[Toy, ...]
      boxes: typing.Mapping[str, Box]

    authors = MaskedTensorV1(
        values=[[b'A', b'Quincy', b'Aardvark'], [b'Z', b'Zhook', b'']],
        mask=[[True, True, True], [True, True, False]])
    toys = [
        Toy('car', 1.0, {
            'size': [8, 3, 2],
            'color': [0.3, 0.2, 0.8]
        }),
        Toy(name='book', price=3.7, features={'authors': authors})
    ]
    boxes = {
        'green': Box(['car']),
        'blue': Box(contents=['car', 'book', 'book'])
    }
    toy_info = ToyInfo(version='1.0 alpha', toys=toys, boxes=boxes)

    @def_function.function
    def fn(info):
      prices = [toy.price for toy in info.toys]
      return math_ops.reduce_sum(array_ops_stack.stack(prices))

    self.assertAllClose(fn(toy_info), 4.7)

  def testNestedCustomConstructor(self):

    class Toy(extension_type.ExtensionType):
      name: str
      price: ops.Tensor

      def __init__(self, name, price, discount=0):
        if discount:
          name += ' (discounted)'
          price *= (1 - discount)
        self.name = name
        self.price = price

    class ToyBox(extension_type.ExtensionType):
      toys: typing.Tuple[Toy, ...]

      def __init__(self, name_to_price, name_to_discount):
        self.toys = [
            Toy(name, price, name_to_discount.get(name, 0))
            for (name, price) in name_to_price.items()
        ]

    toy_box = ToyBox({
        'car': 8.3,
        'truck': 5.9,
        'puzzle': 5.3,
        'jacks': 2.8
    }, {
        'puzzle': .2,
        'truck': .3
    })
    self.assertLen(toy_box.toys, 4)
    self.assertEqual(
        set(toy.name for toy in toy_box.toys),
        {'car', 'truck (discounted)', 'puzzle (discounted)', 'jacks'})

  def testExtensionTypeWithMathOperators(self):

    def masked_add(x, y, name=None):
      del name
      if not isinstance(x, MaskedTensorV2) and isinstance(y, MaskedTensorV2):
        return dispatch.OpDispatcher.NOT_SUPPORTED
      return MaskedTensorV2(x.values + y.values, x.mask & y.mask)

    with temporarily_add_dispatch(math_ops.add, MaskedTensorV2, masked_add):
      x = MaskedTensorV2([[1, 2], [3, 4]], [[True, False], [True, True]])
      y = MaskedTensorV2([[3, 4], [5, 6]], [[True, True], [False, True]])
      z = x + y
      self.assertAllEqual(z.values, [[4, 6], [8, 10]])
      self.assertAllEqual(z.mask, [[True, False], [False, True]])

  def testGetExtensionTypeFields(self):

    # Can be called on a type or an instance:
    fields_1 = MaskedTensorV1._tf_extension_type_fields()
    fields_2 = MaskedTensorV1([0], [True])._tf_extension_type_fields()

    for fields in [fields_1, fields_2]:
      self.assertLen(fields, 2)
      self.assertEqual(fields[0].name, 'values')
      self.assertEqual(fields[0].value_type, ops.Tensor)
      self.assertEqual(fields[0].default, fields[0].NO_DEFAULT)
      self.assertEqual(fields[1].name, 'mask')
      self.assertEqual(fields[1].value_type, ops.Tensor)
      self.assertEqual(fields[1].default, fields[0].NO_DEFAULT)

  def testHasExtensionTypeField(self):

    self.assertTrue(MaskedTensorV1._tf_extension_type_has_field('values'))
    self.assertTrue(MaskedTensorV1._tf_extension_type_has_field('mask'))
    self.assertFalse(MaskedTensorV1._tf_extension_type_has_field('labels'))

    mt = MaskedTensorV1([0], [True])
    self.assertTrue(mt._tf_extension_type_has_field('values'))
    self.assertTrue(mt._tf_extension_type_has_field('mask'))
    self.assertFalse(mt._tf_extension_type_has_field('labels'))

  def testForwardReferences(self):
    A, B = ForwardRefA, ForwardRefB

    self.assertEqual(A._tf_extension_type_fields(),
                     (extension_type_field.ExtensionTypeField(
                         'x', typing.Tuple[typing.Union[A, B], ...]),
                      extension_type_field.ExtensionTypeField('y', B)))
    self.assertEqual(B._tf_extension_type_fields(),
                     (extension_type_field.ExtensionTypeField('z', B),
                      extension_type_field.ExtensionTypeField('n', ops.Tensor)))

    # Check the signature.
    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter(
            'x',
            POSITIONAL_OR_KEYWORD,
            annotation=typing.Tuple[typing.Union['ForwardRefA', 'ForwardRefB'],
                                    ...]),
        tf_inspect.Parameter(
            'y', POSITIONAL_OR_KEYWORD, annotation='ForwardRefB'),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=A)
    self.assertEqual(tf_inspect.signature(A.__init__), expected_sig)

  def testUnresolvedForwardReference(self):

    class Broken(extension_type.ExtensionType):
      x: 'Cra'  # note: intentional typo for Car.

    class Car(extension_type.ExtensionType):
      speed: float

    with self.assertRaises(TypeError):
      Broken(x=Car(3.8))

  def testUnsupportedAnnotations(self):
    with self.assertRaisesRegex(
        TypeError, "In field 'values': Unsupported type annotation"):

      class MyType1(extension_type.ExtensionType):  # pylint: disable=unused-variable
        values: typing.List[ops.Tensor]

    with self.assertRaisesRegex(TypeError,
                                "In field 'xyz': Unsupported type annotation"):

      class MyType2(extension_type.ExtensionType):  # pylint: disable=unused-variable
        xyz: typing.Union[typing.Tuple[complex, ...], int]

  def testCantUseReservedName(self):
    with self.assertRaisesRegex(
        ValueError, 'The field annotations for MyType1 are invalid. '
        "Field '_to_components' is reserved"):

      class MyType1(extension_type.ExtensionType):  # pylint: disable=unused-variable
        _to_components: int

    with self.assertRaisesRegex(
        ValueError, 'The field annotations for MyType2 are invalid. '
        "Field '_tf_extension_type_foo' is reserved"):

      class MyType2(extension_type.ExtensionType):  # pylint: disable=unused-variable
        _tf_extension_type_foo: int

    with self.assertRaisesRegex(
        ValueError, 'The field annotations for MyType3 are invalid. '
        "Field 'is_compatible_with' is reserved"):

      class MyType3(extension_type.ExtensionType):  # pylint: disable=unused-variable

        def is_compatible_with(self, other):
          return False

  def testExtensionTypeBaseClassHasNoSpec(self):
    self.assertFalse(hasattr(extension_type.ExtensionType, 'Spec'))

  def testExtensionTypeBaseConstructorRaisesException(self):
    with self.assertRaisesRegex(AssertionError,
                                'ExtensionType is an abstract base class.'):
      extension_type.ExtensionType()

  class ExtensionTypeWithName(extension_type.ExtensionType):
    __name__ = 'tf.__test__.ExtensionTypeWithName'  # For SavedModel
    x: typing.Tuple[ops.Tensor, int]
    y: ops.Tensor

  def testSavedModelSupport(self):

    class TestModule(module.Module):

      @def_function.function
      def f(self, s):
        return s.x[0] + s.x[1] + s.y

    s1 = self.ExtensionTypeWithName((1, 2), 3)
    s2 = self.ExtensionTypeWithName((1.0, 2), [3.0, 4.0])

    m = TestModule()
    m.f.get_concrete_function(s1)
    m.f.get_concrete_function(s2)

    path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    save.save(m, path)
    loaded = load.load(path)

    self.assertAllEqual(loaded.f(s1), 6)
    self.assertAllEqual(loaded.f(s2), [6.0, 7.0])

  def testPackedEncoding(self):
    mt1 = MaskedTensorV2([1, 2, 3, 4], [True, True, False, True])
    self.assertLen(nest.flatten(mt1, expand_composites=True), 2)

    mt2 = extension_type.pack(mt1)
    self.assertLen(nest.flatten(mt2, expand_composites=True), 1)
    self.assertIsInstance(mt2.values, ops.Tensor)
    self.assertAllEqual(mt2.values, [1, 2, 3, 4])
    self.assertIsInstance(mt2.mask, ops.Tensor)
    self.assertAllEqual(mt2.mask, [True, True, False, True])

    mt3 = extension_type.unpack(mt2)
    self.assertLen(nest.flatten(mt3, expand_composites=True), 2)
    self.assertIsInstance(mt3.values, ops.Tensor)
    self.assertAllEqual(mt3.values, [1, 2, 3, 4])
    self.assertIsInstance(mt3.mask, ops.Tensor)
    self.assertAllEqual(mt3.mask, [True, True, False, True])

    nest.assert_same_structure(mt1, mt3, expand_composites=True)
    with self.assertRaisesRegex(ValueError, "don't have the same"):  # pylint: disable=g-error-prone-assert-raises
      nest.assert_same_structure(mt1, mt2, expand_composites=True)

    mt4 = MaskedTensorV1([1, 2, 3, 4], [True, True, False, True])
    with self.assertRaisesRegex(
        ValueError,
        'ExtensionTypes must have a __name__ field in order to be packed.'):
      extension_type.pack(mt4)

  def testSubclassing(self):

    class Instrument(extension_type.ExtensionType):
      name: ops.Tensor
      weight: ops.Tensor
      needs_case: bool

    class StringInstrument(Instrument):
      num_strings: int  # Add a new field
      needs_case: bool = True  # Override default value.

    class Violin(StringInstrument):
      maker: ops.Tensor
      num_strings: int = 4  # Override default value.
      name: str = 'violin'  # Override field type and default value.

    self.assertEqual(
        list(
            tf_inspect.signature(
                StringInstrument.__init__).parameters.values()), [
                    tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
                    tf_inspect.Parameter(
                        'name', POSITIONAL_OR_KEYWORD, annotation=ops.Tensor),
                    tf_inspect.Parameter(
                        'weight', POSITIONAL_OR_KEYWORD, annotation=ops.Tensor),
                    tf_inspect.Parameter(
                        'needs_case',
                        POSITIONAL_OR_KEYWORD,
                        annotation=bool,
                        default=True),
                    tf_inspect.Parameter(
                        'num_strings', KEYWORD_ONLY, annotation=int),
                ])
    self.assertEqual(
        list(tf_inspect.signature(Violin.__init__).parameters.values()), [
            tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
            tf_inspect.Parameter(
                'name', POSITIONAL_OR_KEYWORD, annotation=str,
                default='violin'),
            tf_inspect.Parameter('weight', KEYWORD_ONLY, annotation=ops.Tensor),
            tf_inspect.Parameter(
                'needs_case', KEYWORD_ONLY, annotation=bool, default=True),
            tf_inspect.Parameter(
                'num_strings', KEYWORD_ONLY, annotation=int, default=4),
            tf_inspect.Parameter('maker', KEYWORD_ONLY, annotation=ops.Tensor),
        ])

    violin = Violin(weight=28, maker='Amati')
    self.assertAllEqual(violin.name, 'violin')
    self.assertAllEqual(violin.weight, 28)
    self.assertAllEqual(violin.needs_case, True)
    self.assertAllEqual(violin.num_strings, 4)
    self.assertAllEqual(violin.maker, 'Amati')


# integration test to test compatibility with high level api like Dataset
# and Keras
class ExtensionTypeIntegrationTest(test_util.TensorFlowTestCase):

  @test_util.run_v2_only
  def testDataset(self):
    mt = MaskedTensorV3([[1], [2], [3]], [[True], [False], [True]])
    ds = dataset_ops.DatasetV2.from_tensors(mt)
    self.assertEqual(next(iter(ds)), mt)

  @test_util.run_v2_only
  def testDatasetBatch(self):
    xs = MaskedTensorV3([[1], [2], [3]], [[True], [False], [True]])
    x0 = MaskedTensorV3(xs.values[0], xs.mask[0])

    ds = dataset_ops.DatasetV2.from_tensors(xs)
    self.assertEqual(next(iter(ds)), xs)
    ds = ds.unbatch()
    self.assertEqual(next(iter(ds)), x0)

    ds = dataset_ops.DatasetV2.from_tensor_slices(xs)
    self.assertEqual(next(iter(ds)), x0)
    ds = ds.batch(3, drop_remainder=True)
    self.assertEqual(next(iter(ds)), xs)

  @test_util.run_v2_only
  def testDatasetBatchRagged(self):
    xs = MaskedTensorV3(
        ragged_factory_ops.constant([[1], [2, 3], [4]]),
        ragged_factory_ops.constant([[True], [False], [True]]))
    x0 = MaskedTensorV3(xs.values[0], xs.mask[0])

    ds = dataset_ops.DatasetV2.from_tensors(xs)
    self.assertEqual(next(iter(ds)), xs)
    ds = ds.unbatch()
    self.assertEqual(next(iter(ds)), x0)

    ds = dataset_ops.DatasetV2.from_tensor_slices(xs)
    self.assertEqual(next(iter(ds)), x0)
    ds = ds.batch(3, drop_remainder=True)
    self.assertEqual(next(iter(ds)), xs)

  @test_util.run_v2_only
  def testDistributedDataset(self):
    strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
    mt = MaskedTensorV3([[1], [2], [3], [4]], [[True], [False], [True], [True]])
    ds = dataset_ops.DatasetV2.from_tensor_slices(mt).batch(2)
    dist_dataset = strategy.experimental_distribute_dataset(ds)
    expect = MaskedTensorV3([[1]], [[True]])
    per_replica_result = next(iter(dist_dataset))
    self.assertEqual(per_replica_result.values[0].values, expect.values[0])
    self.assertEqual(per_replica_result.values[0].mask, expect.mask[0])


@test_util.run_all_in_graph_and_eager_modes
class ExtensionTypeSpecTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):

  def testSpecConstructor(self):
    values_spec = tensor_spec.TensorSpec([4], dtypes.float32)
    mask_spec = tensor_spec.TensorSpec([4], dtypes.bool)
    mt_spec = MaskedTensorV1.Spec(values_spec, mask_spec)
    self.assertEqual(mt_spec.values, values_spec)
    self.assertEqual(mt_spec.mask, mask_spec)

    mt = MaskedTensorV1([1.0, 2.0, 3.0, 4.0], [True, True, False, True])
    self.assertEqual(mt._type_spec, mt_spec)

  def testSpecConstructorSignature(self):

    class MyType(extension_type.ExtensionType):
      x: ops.Tensor
      y: ops.Tensor
      z: typing.Tuple[typing.Union[int, str], ...] = [1, 'two', 3]

    expected_parameters = [
        tf_inspect.Parameter('self', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('x', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('y', POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('z', POSITIONAL_OR_KEYWORD),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyType.Spec)
    self.assertEqual(expected_sig, tf_inspect.signature(MyType.Spec.__init__))

  def testSpecAttributesAreImmutable(self):
    mt = MaskedTensorV1([1, 2, 3, 4], [True, True, False, True])
    mt_spec = MaskedTensorV1.Spec.from_value(mt)
    with self.assertRaisesRegex(
        AttributeError, 'Cannot mutate attribute `score` '
        'outside the custom constructor of ExtensionTypeSpec'):
      mt_spec.score = 12
    with self.assertRaisesRegex(
        AttributeError, 'Cannot mutate attribute `values` '
        'outside the custom constructor of ExtensionTypeSpec'):
      mt_spec.values = constant_op.constant([4, 3, 2, 1])
    with self.assertRaisesRegex(
        AttributeError, 'Cannot mutate attribute `values` '
        'outside the custom constructor of ExtensionTypeSpec'):
      del mt_spec.values

  def testSpecFromValue(self):
    mt = MaskedTensorV1([1.0, 2.0, 3.0, 4.0], [True, True, False, True])
    mt_spec = MaskedTensorV1.Spec.from_value(mt)

    expected_values_spec = tensor_spec.TensorSpec([4], dtypes.float32)
    expected_mask_spec = tensor_spec.TensorSpec([4], dtypes.bool)
    self.assertEqual(mt_spec.values, expected_values_spec)
    self.assertEqual(mt_spec.mask, expected_mask_spec)

  def testSpecSerialize(self):

    class Zoo(extension_type.ExtensionType):
      zookeepers: typing.Tuple[str, ...]
      animals: typing.Mapping[str, typing.Mapping[str, ops.Tensor]]

    featurespec = {
        'size': tensor_spec.TensorSpec([3]),
        'weight': tensor_spec.TensorSpec([])
    }
    zoo_spec = Zoo.Spec(
        zookeepers=['Zoey', 'Zack'],
        animals={
            'tiger': featurespec,
            'elephant': featurespec
        })

    serialized = zoo_spec._serialize()
    self.assertEqual(serialized,
                     (('zookeepers', ('Zoey', 'Zack')), ('animals', {
                         'tiger': featurespec,
                         'elephant': featurespec
                     })))
    restored = Zoo.Spec._deserialize(serialized)
    self.assertEqual(zoo_spec, restored)

    # ImmutableDict is used for the field, but dict for the serialization:
    self.assertIsInstance(zoo_spec.animals, immutable_dict.ImmutableDict)
    serialized_field_name, serialized_field_value = serialized[1]
    self.assertEqual(serialized_field_name, 'animals')
    self.assertIsInstance(serialized_field_value, dict)

  def testSpecComponents(self):

    class Zoo(extension_type.ExtensionType):
      zookeepers: typing.Tuple[str, ...]
      animals: typing.Mapping[str, typing.Mapping[str, ops.Tensor]]

    zoo = Zoo(
        ['Zoey', 'Zack'], {
            'elephant': {
                'size': [25, 30, 20],
                'weight': 2000.0
            },
            'tiger': {
                'hunger': 3.2,
                'size': [3, 8, 2],
                'weight': 87.3
            }
        })
    zoo_spec = Zoo.Spec.from_value(zoo)

    components = zoo_spec._to_components(zoo)
    self.assertLen(components, 5)
    self.assertAllClose(components[0], [25, 30, 20])
    self.assertAllClose(components[1], 2000.0)
    self.assertAllClose(components[2], 3.2)
    self.assertAllClose(components[3], [3, 8, 2])
    self.assertAllClose(components[4], 87.3)

    restored = zoo_spec._from_components(components)
    self.assertAllEqual(zoo == restored, True)

    self.assertEqual(zoo_spec._component_specs,
                     (tensor_spec.TensorSpec([3], dtypes.int32),
                      tensor_spec.TensorSpec([], dtypes.float32),
                      tensor_spec.TensorSpec([], dtypes.float32),
                      tensor_spec.TensorSpec([3], dtypes.int32),
                      tensor_spec.TensorSpec([], dtypes.float32)))

  def testCopyAndPickle(self):
    values_spec = tensor_spec.TensorSpec([4], dtypes.float32)
    mask_spec = tensor_spec.TensorSpec([4], dtypes.bool)
    mt_spec = MaskedTensorV1.Spec(values_spec, mask_spec)
    self.assertEqual(copy.copy(mt_spec), mt_spec)
    self.assertEqual(copy.deepcopy(mt_spec), mt_spec)
    self.assertEqual(pickle.loads(pickle.dumps(mt_spec)), mt_spec)

  def testCustomizeSpecTest(self):

    class WeightedTensor(extension_type.ExtensionType):
      """ExtensionType with a customized TypeSpec.

      * Custom constructor.
      * Custom __validate__.
      * Add properties (shape, dtype, weight_dtype).
      * Add method (with_shape).
      """
      values: ops.Tensor
      weight: ops.Tensor  # scalar

      shape = property(lambda self: self.shape)
      dtype = property(lambda self: self.dtype)
      weight_dtype = property(lambda self: self.weight.dtype)

      def __validate__(self):
        self.weight.shape.assert_has_rank(0)

      class Spec:

        def __init__(self, shape, dtype, weight_dtype=dtypes.float32):
          self.values = tensor_spec.TensorSpec(shape, dtype)
          self.weight = tensor_spec.TensorSpec([], weight_dtype)

        def __validate__(self):
          self.weight.shape.assert_has_rank(0)

        shape = property(lambda self: self.values.shape)
        dtype = property(lambda self: self.values.dtype)
        weight_dtype = property(lambda self: self.weight.dtype)

        def with_shape(self, shape):
          return WeightedTensor.Spec(shape, self.dtype, self.weight_dtype)

    wt = WeightedTensor([1, 2], 0.3)
    wt_spec = WeightedTensor.Spec.from_value(wt)
    self.assertEqual(wt_spec.shape, tensor_shape.TensorShape([2]))
    self.assertEqual(wt_spec.dtype, dtypes.int32)

    self.assertEqual(wt_spec, WeightedTensor.Spec([2], dtypes.int32))

    wt2 = WeightedTensor([[1, 2], [3, 4]], 0.5)
    wt2_spec = WeightedTensor.Spec.from_value(wt2)
    self.assertEqual(wt_spec.with_shape([2, 2]), wt2_spec)

  def testNestedSpecMustBeAClass(self):
    with self.assertRaisesRegex(
        ValueError,
        r'BrokenExtensionType\.Spec must be a nested class; got 12.'):

      class BrokenExtensionType(extension_type.ExtensionType):

        Spec = 12  # pylint: disable=invalid-name

      del BrokenExtensionType

  def testNestedSpecMayNotHaveBaseClasses(self):
    with self.assertRaisesRegex(
        ValueError, r'BrokenExtensionType\.Spec must be directly subclassed '
        'from tf.TypeSpec.'):

      class BrokenExtensionType(extension_type.ExtensionType):

        class Spec(type_spec.BatchableTypeSpec):
          pass

      del BrokenExtensionType


@test_util.run_all_in_graph_and_eager_modes
class AnonymousExtensionTypeTest(test_util.TensorFlowTestCase,
                                 parameterized.TestCase):

  @parameterized.parameters([
      [dict(i=5, f=3.2, b=True, n=None)],
      [dict(x=(1, 2), y={
          3: 4,
          5: 6
      })],
      [lambda: dict(t=constant_op.constant(123))],
      [lambda: dict(r=ragged_factory_ops.constant([[1, 2], [3]]))],
  ])
  def testConstruction(self, fields):
    if callable(fields):
      fields = fields()
    extension_type.AnonymousExtensionType(**fields)

  @parameterized.parameters([
      [dict(x=[1, 2, 3]), 'unsupported `value` argument'],
      [dict(x=set([1, 2])), 'unsupported `value` argument'],
      [dict(x=(1, dict([(2, [])]))), 'unsupported `value` argument'],
      [
          dict(_tf_extension_type_xyz=5),
          'Reserved field name .*_tf_extension_type_xyz.*'
      ],
  ])
  def testConstructionErrors(self, fields, error):
    with self.assertRaisesRegex(ValueError, error):
      extension_type.AnonymousExtensionType(**fields)

  @parameterized.parameters([
      [dict(i=5, f=3.2, b=True, n=None)],
      [dict(x=(1, 2), y={
          3: 4,
          5: 6
      })],
      [lambda: dict(t=constant_op.constant(123))],
      [lambda: dict(r=ragged_factory_ops.constant([[1, 2], [3]]))],
  ])
  def testAttributeAccessors(self, fields):
    if callable(fields):
      fields = fields()
    s = extension_type.AnonymousExtensionType(**fields)
    for (name, value) in fields.items():
      actual = getattr(s, name)
      if isinstance(actual, (ops.Tensor, ragged_tensor.RaggedTensor)):
        self.assertAllEqual(actual, value)
      else:
        self.assertEqual(actual, value)

  def testAttributeAccessorsAreImmutable(self):
    s = extension_type.AnonymousExtensionType(x=12, y={'x': 55})
    with self.assertRaisesRegex(AttributeError, 'Cannot set attribute `x`'):
      s.x = 22
    with self.assertRaisesRegex(AttributeError, 'Cannot delete attribute `y`'):
      del s.y
    with self.assertRaisesRegex(TypeError, 'does not support item assignment'):
      s.y['x'] = 66

  def testReinterpret(self):
    x = MaskedTensorV2([4, 5], [True, False])
    anon_x = extension_type.reinterpret(x,
                                        extension_type.AnonymousExtensionType)
    self.assertAllEqual(anon_x.values, [4, 5])
    self.assertAllEqual(anon_x.mask, [True, False])

    round_trip_x = extension_type.reinterpret(anon_x, MaskedTensorV2)
    self.assertAllEqual(round_trip_x.values, [4, 5])
    self.assertAllEqual(round_trip_x.mask, [True, False])

    converted_x = extension_type.reinterpret(anon_x, MaskedTensorV1)
    self.assertAllEqual(converted_x.values, [4, 5])
    self.assertAllEqual(converted_x.mask, [True, False])

  # pylint: disable=g-long-lambda
  @parameterized.parameters([
      [
          lambda: extension_type.AnonymousExtensionType(
              values=constant_op.constant([1, 2, 3])), MaskedTensorV2,
          "Missing required fields: {'mask'}"
      ],
      [
          lambda: extension_type.AnonymousExtensionType(
              values=(1, 2, 3), mask=None), MaskedTensorV2,
          "mask: expected a Tensor, got 'NoneType'"
      ],
      [
          lambda: extension_type.AnonymousExtensionType(
              values=constant_op.constant([1, 2, 3]),
              mask=constant_op.constant([True, False])), MaskedTensorV2,
          'Shapes .* are incompatible'
      ],
      [
          lambda: extension_type.AnonymousExtensionType(
              values=constant_op.constant([1, 2, 3])), ops.Tensor,
          'reinterpret expects `new_type` to be a subclass of '
          'tf.ExtensionType; '
          'got .*.Tensor.*'
      ],
      [
          lambda: constant_op.constant([1, 2, 3]),
          extension_type.AnonymousExtensionType,
          'reinterpret expects `value` to be a tf.ExtensionType instance; '
          'got.*.Tensor.*'
      ],
  ])
  def testReinterpretErrors(self, value, new_type, error):
    if callable(value):
      value = value()
    with self.assertRaisesRegex((TypeError, ValueError), error):
      extension_type.reinterpret(value, new_type)

  def testLoadSavedModelWithUnregisteredExtensionType(self):

    def f(x, y):
      x_values = x.values if isinstance(x, MaskedTensorV1) else x
      y_values = y.values if isinstance(y, MaskedTensorV1) else y
      x_mask = x.mask if isinstance(x, MaskedTensorV1) else True
      y_mask = y.mask if isinstance(y, MaskedTensorV1) else True
      return MaskedTensorV1(x_values + y_values, x_mask & y_mask)

    t_spec = tensor_spec.TensorSpec(None, dtypes.int32)
    b_spec = tensor_spec.TensorSpec(None, dtypes.bool)
    mt_spec = MaskedTensorV1.Spec(values=t_spec, mask=b_spec)
    model = module.Module()
    model.f = def_function.function(f)
    model.f.get_concrete_function(t_spec, t_spec)
    model.f.get_concrete_function(t_spec, mt_spec)
    model.f.get_concrete_function(mt_spec, t_spec)
    model.f.get_concrete_function(mt_spec, mt_spec)

    path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    with temporarily_register_type_spec('tf.test.MaskedTensorV1.Spec',
                                        MaskedTensorV1.Spec):
      save.save(model, path)
    loaded_model = load.load(path)

    with self.assertRaises(ValueError):
      type_spec_registry.lookup('tf.test.MaskedTensorV1')

    t = constant_op.constant([10, 20, 30])
    v1 = loaded_model.f(t, t)
    self.assertIsInstance(v1, extension_type.AnonymousExtensionType)
    self.assertAllEqual(v1.values, [20, 40, 60])
    self.assertAllEqual(v1.mask, True)

    v2 = loaded_model.f(v1, v1)
    self.assertIsInstance(v2, extension_type.AnonymousExtensionType)
    self.assertAllEqual(v2.values, [40, 80, 120])
    self.assertAllEqual(v2.mask, True)

    mt = MaskedTensorV1([1, 2, 3], [True, True, False])
    v3 = loaded_model.f(
        t, extension_type.reinterpret(mt,
                                      extension_type.AnonymousExtensionType))
    self.assertIsInstance(v3, extension_type.AnonymousExtensionType)
    self.assertAllEqual(v3.values, [11, 22, 33])
    self.assertAllEqual(v3.mask, [True, True, False])

    v4 = extension_type.reinterpret(v3, MaskedTensorV1)
    self.assertIsInstance(v4, MaskedTensorV1)
    self.assertAllEqual(v4.values, [11, 22, 33])
    self.assertAllEqual(v4.mask, [True, True, False])

  def testFlatTensorSpecs(self):
    x = MaskedTensorV2([4, 5], [True, False])
    spec = type_spec.type_spec_from_value(x)
    flat_specs = spec._flat_tensor_specs
    self.assertEqual(flat_specs, [
        tensor_spec.TensorSpec(shape=(2,), dtype=dtypes.int32, name=None),
        tensor_spec.TensorSpec(shape=(2,), dtype=dtypes.bool, name=None)
    ])

  def testFullTypesForFlatTensors(self):
    x = MaskedTensorV2([4, 5], [True, False])
    spec = type_spec.type_spec_from_value(x)
    full_type_list = fulltypes_for_flat_tensors(spec)
    expect = [
        full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET),
        full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET)
    ]
    self.assertEqual(len(spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)


def replace_tensors_with_placeholders(value):

  def repl(x):
    if isinstance(x, ops.Tensor):
      return array_ops.placeholder_with_default(x, shape=None)
    else:
      return x

  return nest.map_structure(repl, value, expand_composites=True)


@contextlib.contextmanager
def temporarily_add_dispatch(op, typ, fn):
  n = len(op._tf_fallback_dispatchers)
  dispatch.dispatch_for_types(op, typ)(fn)
  yield
  assert len(op._tf_fallback_dispatchers) == n + 1
  del op._tf_fallback_dispatchers[-1]


@contextlib.contextmanager
def temporarily_register_type_spec(name, cls):
  """Context manager for making temporary changes to the TypeSpec registry."""
  type_spec_registry.register(name)(cls)
  yield
  assert type_spec_registry._TYPE_SPEC_TO_NAME.pop(cls) == name
  assert type_spec_registry._NAME_TO_TYPE_SPEC.pop(name) is cls


if __name__ == '__main__':
  googletest.main()
