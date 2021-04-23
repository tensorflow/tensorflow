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
"""Tests for tf.framework.tensor_struct."""

import contextlib
import typing

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import struct_field
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_struct
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect


def build_simple_masked_tensor_type():

  class MaskedTensor(tensor_struct.Struct):
    values: ops.Tensor
    mask: tensor_spec.TensorSpec(shape=None, dtype=dtypes.bool)

  return MaskedTensor


def build_advanced_masked_tensor_type():

  def _masked_tensor_repr(values, mask):
    assert len(values) == len(mask)
    if len(values.shape) == 1:
      items = [repr(v) if m else '_' for (v, m) in zip(values, mask)]
    else:
      items = [_masked_tensor_repr(v, m) for (v, m) in zip(values, mask)]
    return '[%s]' % ', '.join(items)

  class MaskedTensor(tensor_struct.Struct):
    values: ops.Tensor
    mask: tensor_spec.TensorSpec(shape=None, dtype=dtypes.bool)

    def __repr__(self):
      if hasattr(self.values, 'numpy') and hasattr(self.mask, 'numpy'):
        return '<MaskedTensor %s>' % _masked_tensor_repr(
            self.values.numpy(), self.mask.numpy())
      else:
        return super(MaskedTensor, self).__repr__()

    @property
    def shape(self):
      return self.values.shape

    @property
    def dtype(self):
      return self.values.dtype

    def __validate__(self):
      self.values.shape.assert_is_compatible_with(self.mask.shape)

    def with_default(self, default):
      return array_ops.where_v2(self.mask, self.values, default)

    __add__ = math_ops.add
    __sub__ = math_ops.subtract

  return MaskedTensor


class ForwardRefA(tensor_struct.Struct):
  x: typing.Tuple[typing.Union['ForwardRefA', 'ForwardRefB'], ...]
  y: 'ForwardRefB'


class ForwardRefB(tensor_struct.Struct):
  z: 'ForwardRefB'
  n: ops.Tensor


@test_util.run_all_in_graph_and_eager_modes
class StructTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testAttributeAccessors(self):
    MaskedTensor = build_simple_masked_tensor_type()
    mt = MaskedTensor([1, 2, 3, 4], [True, True, False, True])
    self.assertIsInstance(mt.values, ops.Tensor)
    self.assertAllEqual(mt.values, [1, 2, 3, 4])
    self.assertIsInstance(mt.mask, ops.Tensor)
    self.assertAllEqual(mt.mask, [True, True, False, True])

  def testAttributesAreImmutable(self):
    MaskedTensor = build_simple_masked_tensor_type()
    mt = MaskedTensor([1, 2, 3, 4], [True, True, False, True])
    with self.assertRaisesRegex(AttributeError,
                                "cannot assign to field 'score'"):
      mt.score = 12
    with self.assertRaisesRegex(AttributeError,
                                "cannot assign to field 'values'"):
      mt.values = constant_op.constant([4, 3, 2, 1])
    with self.assertRaisesRegex(AttributeError, "cannot delete field 'values'"):
      del mt.values

  def testRepr(self):
    MaskedTensor = build_simple_masked_tensor_type()
    values = constant_op.constant([1, 2, 3, 4])
    mask = constant_op.constant([True, True, False, True])
    mt = MaskedTensor(values, mask)
    expected = f'MaskedTensor(values={values!r}, mask={mask!r})'
    self.assertEqual(expected, repr(mt))

  def testEagerRepr(self):
    MaskedTensor = build_advanced_masked_tensor_type()
    values = constant_op.constant([1, 2, 3, 4])
    mask = constant_op.constant([True, True, False, True])
    mt = MaskedTensor(values, mask)
    if context.executing_eagerly():
      expected = '<MaskedTensor [1, 2, _, 4]>'
    else:
      expected = f'MaskedTensor(values={values!r}, mask={mask!r})'

    self.assertEqual(expected, repr(mt))
    self.assertEqual(expected, repr(mt))

  def testConstructorSignature(self):

    class MyStruct(tensor_struct.Struct):
      x: ops.Tensor
      y: tensor_spec.TensorSpec(shape=None, dtype=dtypes.bool)
      z: typing.Tuple[typing.Union[int, str], ...] = [1, 'two', 3]

    expected_parameters = [
        tf_inspect.Parameter('self',
                             tf_inspect.Parameter.POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter(
            'x',
            tf_inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=ops.Tensor),
        tf_inspect.Parameter(
            'y',
            tf_inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=tensor_spec.TensorSpec(shape=None, dtype=dtypes.bool)),
        tf_inspect.Parameter(
            'z',
            tf_inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=typing.Tuple[typing.Union[int, str], ...],
            default=(1, 'two', 3)),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyStruct)
    self.assertEqual(expected_sig, tf_inspect.signature(MyStruct.__init__))

  def testEmptyStruct(self):

    class EmptyStruct(tensor_struct.Struct):
      pass

    self.assertEmpty(EmptyStruct._tf_struct_fields())
    x = EmptyStruct()
    self.assertEqual(repr(x), 'EmptyStruct()')

  def testCustomConstrutor(self):

    class SummarizedTensor(tensor_struct.Struct):
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

  def testCustomConstrutorCantMutateNestedValues(self):

    class Foo(tensor_struct.Struct):
      x: int

    class Bar(tensor_struct.Struct):
      foo: Foo

      def __init__(self, foo):
        foo.x = 33  # This raises an exception

    with self.assertRaisesRegex(AttributeError, "cannot assign to field 'x'"):
      Bar(Foo(12))

  def testCustomValidate(self):

    class AlignedTensors(tensor_struct.Struct):
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

    class MyStruct(tensor_struct.Struct):
      values: ops.Tensor
      score: ops.Tensor
      flavor: str

    x1 = MyStruct([1, 2], 8, 'blue')
    x2 = MyStruct([1, 2], 8, 'blue')
    y = MyStruct([1, 2], 8, 'red')
    z = MyStruct([1, 2], 7, 'blue')
    self.assertAllEqual(x1 == x2, True)
    self.assertAllEqual(x1 != x2, False)
    self.assertAllEqual(x1 == y, False)
    self.assertAllEqual(x1 != y, True)
    self.assertAllEqual(x1 == z, False)
    self.assertAllEqual(y == z, False)

    # These are not equal, even though their values are broadcast-compatible
    # and elements are all equal when we broadcast.  Shapes must match.
    a = MyStruct([1, 1, 1, 1], 0, 'x')
    b = MyStruct([[1, 1, 1, 1]], 0, 'x')
    c = MyStruct([[1, 1], [1, 1]], 0, 'x')
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
    MaskedTensor = build_advanced_masked_tensor_type()

    @def_function.function
    def fn(x):
      return x.with_default(99)

    mt = MaskedTensor([1, 2, 3, 4], [True, True, False, True])
    self.assertAllEqual([1, 2, 99, 4], fn(mt))

  def testReturnFromTfFunction(self):
    MaskedTensor = build_advanced_masked_tensor_type()

    @def_function.function
    def mask_neg_values(x):
      return MaskedTensor(x, x > 0)

    actual = mask_neg_values(constant_op.constant([5, 8, -3, 9]))
    expected = MaskedTensor([5, 8, -3, 9], [True, True, False, True])
    self.assertIsInstance(actual, MaskedTensor)
    self.assertAllEqual(expected.values, actual.values)
    self.assertAllEqual(expected.mask, actual.mask)

  def testCaptureByTfFunction(self):
    MaskedTensor = build_advanced_masked_tensor_type()

    x = MaskedTensor(
        values=[[1, 2, 3], [4, 5, 6]],
        mask=[[True, True, True], [True, False, True]])

    @def_function.function
    def add_to_x(y):
      return MaskedTensor(x.values + y.values, x.mask & y.mask)

    actual = add_to_x(MaskedTensor([10, 20, 30], [False, True, True]))
    expected = MaskedTensor(
        values=[[11, 22, 33], [14, 25, 36]],
        mask=[[False, True, True], [False, False, True]])
    self.assertIsInstance(actual, MaskedTensor)
    self.assertAllEqual(expected.values, actual.values)
    self.assertAllEqual(expected.mask, actual.mask)

  def testTfFunctionArgMutationError(self):
    MaskedTensor = build_simple_masked_tensor_type()

    @def_function.function
    def fn_with_side_effect(mts):
      mts.append(MaskedTensor(mts[0].values * 2, mts[0].mask))

    with self.assertRaisesRegex(ValueError, 'should not modify'):
      fn_with_side_effect([MaskedTensor([10, 20, 30], [False, True, True])])

  def testNestPackUnpack(self):

    class CandyStore(tensor_struct.Struct):
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
    MaskedTensor = build_simple_masked_tensor_type()
    x = MaskedTensor([1, 2, 3, 4], [True, False, True, False])
    y = MaskedTensor([5, 6, 7, 8], [False, True, True, False])

    x_2 = control_flow_ops.cond(
        constant_op.constant(True), lambda: x, lambda: y)
    y_2 = control_flow_ops.cond(
        constant_op.constant(False), lambda: x, lambda: y)

    self.assertAllEqual(x.values, x_2.values)
    self.assertAllEqual(x.mask, x_2.mask)
    self.assertAllEqual(y.values, y_2.values)
    self.assertAllEqual(y.mask, y_2.mask)

  def testComplexCond(self):
    MaskedTensor = build_simple_masked_tensor_type()

    mt = MaskedTensor([1, 2, 3, 4], [True, False, True, False])

    def true_fn():
      return MaskedTensor(
          array_ops.where_v2(mt.mask, mt.values, -1), mt.values > 3)

    def false_fn():
      return MaskedTensor(
          array_ops.where_v2(mt.mask, 100, mt.values * 2),
          math_ops.logical_not(mt.mask))

    x = control_flow_ops.cond(constant_op.constant(True), true_fn, false_fn)
    y = control_flow_ops.cond(constant_op.constant(False), true_fn, false_fn)

    self.assertAllEqual(x.values, [1, -1, 3, -1])
    self.assertAllEqual(x.mask, [False, False, False, True])
    self.assertAllEqual(y.values, [100, 4, 100, 8])
    self.assertAllEqual(y.mask, [False, True, False, True])

  def testCondAutograph(self):
    MaskedTensor = build_simple_masked_tensor_type()

    @def_function.function
    def fn(mt):
      if mt.values[3] > 3:
        return MaskedTensor(
            array_ops.where_v2(mt.mask, mt.values, -1), mt.values > 3)
      else:
        return MaskedTensor(
            array_ops.where_v2(mt.mask, 100, mt.values * 2), not mt.mask)

    x = fn(MaskedTensor([1, 2, 3, 4], [True, False, True, False]))
    self.assertAllEqual(x.values, [1, -1, 3, -1])
    self.assertAllEqual(x.mask, [False, False, False, True])

  def testCondTypeMismatch(self):
    if context.executing_eagerly:
      # In eager mode, tf.cond eagerly runs either true_fn or false_fn, and
      # ignores the other one; so it doesn't detect any type mismatches
      # between the two outcomes.  (See _eager_cond_implementation in
      # control_flow_ops.py.)
      return

    class MaskedTensor(tensor_struct.Struct):
      values: ops.Tensor
      mask: tensor_spec.TensorSpec(shape=None, dtype=dtypes.bool)

    class MaskedTensorV2(tensor_struct.Struct):
      values: ops.Tensor
      mask: tensor_spec.TensorSpec(shape=None, dtype=dtypes.bool)

    a = lambda: MaskedTensor([1, 2, 3], [True, True, False])
    b = lambda: MaskedTensor(['a', 'b', 'c'], [False, True, True])
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

  def testWhileLoop(self):
    MaskedTensor = build_simple_masked_tensor_type()
    x = MaskedTensor([1, 2, 3, 4], [True, False, True, False])

    cond = lambda i, x: i < 10
    body = lambda i, x: (i + 1, MaskedTensor(x.values * 2, x.mask))
    _, y = control_flow_ops.while_loop_v2(cond, body, [0, x])

    self.assertIsInstance(y, MaskedTensor)
    self.assertAllEqual(y.values, [1024, 2048, 3072, 4096])
    self.assertAllEqual(y.mask, [True, False, True, False])

  def testWhileLoopAutograph(self):
    MaskedTensor = build_simple_masked_tensor_type()

    @def_function.function
    def fn(x, n):
      for _ in math_ops.range(n):
        x = MaskedTensor(x.values * 2, x.mask)
      return x

    y = fn(MaskedTensor([1, 2, 3, 4], [True, False, True, False]), 10)
    self.assertIsInstance(y, MaskedTensor)
    self.assertAllEqual(y.values, [1024, 2048, 3072, 4096])
    self.assertAllEqual(y.mask, [True, False, True, False])

  def testWhileLoopTypeMismatch(self):
    MaskedTensor = build_simple_masked_tensor_type()
    x = MaskedTensor([1, 2, 3, 4], [True, False, True, False])

    cond = lambda i, x: i < 10

    def body(i, x):
      if isinstance(x, MaskedTensor):
        return x.values * 2
      else:
        return MaskedTensor(x, x > i)

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure"):
      control_flow_ops.while_loop_v2(cond, body, [0, x])

  def testNestedFields(self):
    PossiblyRaggedTensor = typing.Union[ops.Tensor, ragged_tensor.RaggedTensor]
    ToyFeatures = typing.Mapping[str, PossiblyRaggedTensor]

    class ToyInfo(tensor_struct.Struct):
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

    if context.executing_eagerly():
      expected_repr = (
          "ToyInfo(version='1.0 alpha', "
          "toys=(('car', <tf.Tensor: shape=(), dtype=float32, numpy=1.0>, "
          'ImmutableDict('
          "{'size': <tf.Tensor: shape=(3,), dtype=int32, "
          'numpy=array([8, 3, 2], dtype=int32)>, '
          "'color': <tf.Tensor: shape=(3,), dtype=float32, "
          'numpy=array([0.3, 0.2, 0.8], dtype=float32)>})), '
          "('book', <tf.Tensor: shape=(), dtype=float32, numpy=3.7>, "
          'ImmutableDict('
          "{'authors': <tf.RaggedTensor [[b'A', b'Aardvark'], "
          "[b'Z', b'Zhook']]>}))), "
          'boxes=ImmutableDict('
          "{'green': <tf.Tensor: shape=(1,), dtype=string, "
          "numpy=array([b'car'], dtype=object)>, "
          "'blue': <tf.Tensor: shape=(3,), "
          "dtype=string, numpy=array([b'car', b'book', b'book'], "
          'dtype=object)>}))')
    else:
      expected_repr = (
          "ToyInfo(version='1.0 alpha', "
          "toys=(('car', <tf.Tensor 'Const:0' shape=() dtype=float32>, "
          'ImmutableDict('
          "{'size': <tf.Tensor 'Const_1:0' shape=(3,) dtype=int32>, "
          "'color': <tf.Tensor 'Const_2:0' shape=(3,) dtype=float32>})), "
          "('book', <tf.Tensor 'Const_3:0' shape=() dtype=float32>, "
          'ImmutableDict('
          "{'authors': tf.RaggedTensor(values=Tensor("
          '"RaggedConstant/values:0", shape=(4,), dtype=string), '
          'row_splits=Tensor("RaggedConstant/Const:0", shape=(3,), '
          'dtype=int64))}))), '
          'boxes=ImmutableDict({'
          "'green': <tf.Tensor 'Const_4:0' shape=(1,) dtype=string>, "
          "'blue': <tf.Tensor 'Const_5:0' shape=(3,) dtype=string>}))")

    self.assertEqual(repr(toy_info), expected_repr)

  def testNestedStructs(self):
    MaskedTensor = build_simple_masked_tensor_type()
    PossiblyMaskedTensor = typing.Union[ops.Tensor, MaskedTensor]

    class Toy(tensor_struct.Struct):
      name: str
      price: ops.Tensor
      features: typing.Mapping[str, PossiblyMaskedTensor]

    class Box(tensor_struct.Struct):
      contents: ops.Tensor

    class ToyInfo(tensor_struct.Struct):
      version: str
      toys: typing.Tuple[Toy, ...]
      boxes: typing.Mapping[str, Box]

    authors = MaskedTensor(
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
      return math_ops.reduce_sum(array_ops.stack(prices))

    self.assertAllClose(fn(toy_info), 4.7)

  def testNestedCustomConstructor(self):

    class Toy(tensor_struct.Struct):
      name: str
      price: ops.Tensor

      def __init__(self, name, price, discount=0):
        if discount:
          name += ' (discounted)'
          price *= (1 - discount)
        self.name = name
        self.price = price

    class ToyBox(tensor_struct.Struct):
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

  def testStructWithMathOperators(self):
    MaskedTensor = build_advanced_masked_tensor_type()

    def masked_add(x, y, name=None):
      del name
      if not isinstance(x, MaskedTensor) and isinstance(y, MaskedTensor):
        return dispatch.OpDispatcher.NOT_SUPPORTED
      return MaskedTensor(x.values + y.values, x.mask & y.mask)

    with temporarily_add_dispatch(math_ops.add, MaskedTensor, masked_add):
      x = MaskedTensor([[1, 2], [3, 4]], [[True, False], [True, True]])
      y = MaskedTensor([[3, 4], [5, 6]], [[True, True], [False, True]])
      z = x + y
      self.assertAllEqual(z.values, [[4, 6], [8, 10]])
      self.assertAllEqual(z.mask, [[True, False], [False, True]])

  def testGetStructFields(self):
    MaskedTensor = build_simple_masked_tensor_type()

    # Can be called on a type or an instance:
    fields_1 = MaskedTensor._tf_struct_fields()
    fields_2 = MaskedTensor([0], [True])._tf_struct_fields()

    for fields in [fields_1, fields_2]:
      self.assertLen(fields, 2)
      self.assertEqual(fields[0].name, 'values')
      self.assertEqual(fields[0].value_type, ops.Tensor)
      self.assertEqual(fields[0].default, fields[0].NO_DEFAULT)
      self.assertEqual(fields[1].name, 'mask')
      self.assertEqual(fields[1].value_type,
                       tensor_spec.TensorSpec(shape=None, dtype=dtypes.bool))
      self.assertEqual(fields[1].default, fields[0].NO_DEFAULT)

  def testHasStructField(self):
    MaskedTensor = build_simple_masked_tensor_type()

    self.assertTrue(MaskedTensor._tf_struct_has_field('values'))
    self.assertTrue(MaskedTensor._tf_struct_has_field('mask'))
    self.assertFalse(MaskedTensor._tf_struct_has_field('labels'))

    mt = MaskedTensor([0], [True])
    self.assertTrue(mt._tf_struct_has_field('values'))
    self.assertTrue(mt._tf_struct_has_field('mask'))
    self.assertFalse(mt._tf_struct_has_field('labels'))

  def testForwardReferences(self):
    A, B = ForwardRefA, ForwardRefB

    self.assertEqual(
        A._tf_struct_fields(),
        (struct_field.StructField('x', typing.Tuple[typing.Union[A, B], ...]),
         struct_field.StructField('y', B)))
    self.assertEqual(B._tf_struct_fields(), (struct_field.StructField(
        'z', B), struct_field.StructField('n', ops.Tensor)))

    # Check the signature.
    expected_parameters = [
        tf_inspect.Parameter('self',
                             tf_inspect.Parameter.POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter(
            'x',
            tf_inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=typing.Tuple[typing.Union['ForwardRefA', 'ForwardRefB'],
                                    ...]),
        tf_inspect.Parameter(
            'y',
            tf_inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation='ForwardRefB'),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=A)
    self.assertEqual(tf_inspect.signature(A.__init__), expected_sig)

  def testUnresolvedForwardReference(self):

    class Broken(tensor_struct.Struct):
      x: 'Cra'  # note: intentional typo for Car.

    class Car(tensor_struct.Struct):
      speed: float

    with self.assertRaises(TypeError):
      Broken(x=Car(3.8))

  def testUnsupportedAnnotations(self):
    with self.assertRaisesRegex(
        TypeError, "In field 'values': Unsupported type annotation"):

      class MyStruct1(tensor_struct.Struct):  # pylint: disable=unused-variable
        values: typing.List[ops.Tensor]

    with self.assertRaisesRegex(TypeError,
                                "In field 'xyz': Unsupported type annotation"):

      class MyStruct2(tensor_struct.Struct):  # pylint: disable=unused-variable
        xyz: typing.Union[typing.Tuple[complex, ...], int]

  def testStructBaseClassHasNoSpec(self):
    self.assertFalse(hasattr(tensor_struct.Struct, 'Spec'))

  def testStructBaseConstructorRaisesException(self):
    with self.assertRaisesRegex(AssertionError,
                                'Struct is an abstract base class.'):
      tensor_struct.Struct()


@test_util.run_all_in_graph_and_eager_modes
class StructSpecTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testSpecConstructor(self):
    MaskedTensor = build_simple_masked_tensor_type()
    values_spec = tensor_spec.TensorSpec([4], dtypes.float32)
    mask_spec = tensor_spec.TensorSpec([4], dtypes.bool)
    mt_spec = MaskedTensor.Spec(values_spec, mask_spec)
    self.assertEqual(mt_spec.values, values_spec)
    self.assertEqual(mt_spec.mask, mask_spec)

    mt = MaskedTensor([1.0, 2.0, 3.0, 4.0], [True, True, False, True])
    self.assertEqual(mt._type_spec, mt_spec)

  def testSpecConstructorSignature(self):

    class MyStruct(tensor_struct.Struct):
      x: ops.Tensor
      y: tensor_spec.TensorSpec(shape=None, dtype=dtypes.bool)
      z: typing.Tuple[typing.Union[int, str], ...] = [1, 'two', 3]

    expected_parameters = [
        tf_inspect.Parameter('self',
                             tf_inspect.Parameter.POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('x', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('y', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD),
        tf_inspect.Parameter('z', tf_inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
    expected_sig = tf_inspect.Signature(
        expected_parameters, return_annotation=MyStruct.Spec)
    self.assertEqual(expected_sig, tf_inspect.signature(MyStruct.Spec.__init__))

  def testSpecAttributesAreImmutable(self):
    MaskedTensor = build_simple_masked_tensor_type()
    mt = MaskedTensor([1, 2, 3, 4], [True, True, False, True])
    mt_spec = MaskedTensor.Spec.from_value(mt)
    with self.assertRaisesRegex(AttributeError,
                                "cannot assign to field 'score'"):
      mt_spec.score = 12
    with self.assertRaisesRegex(AttributeError,
                                "cannot assign to field 'values'"):
      mt_spec.values = constant_op.constant([4, 3, 2, 1])
    with self.assertRaisesRegex(AttributeError, "cannot delete field 'values'"):
      del mt_spec.values

  def testSpecFromValue(self):
    MaskedTensor = build_simple_masked_tensor_type()
    mt = MaskedTensor([1.0, 2.0, 3.0, 4.0], [True, True, False, True])
    mt_spec = MaskedTensor.Spec.from_value(mt)

    expected_values_spec = tensor_spec.TensorSpec([4], dtypes.float32)
    expected_mask_spec = tensor_spec.TensorSpec([4], dtypes.bool)
    self.assertEqual(mt_spec.values, expected_values_spec)
    self.assertEqual(mt_spec.mask, expected_mask_spec)

  def testSpecSerialize(self):

    class Zoo(tensor_struct.Struct):
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
    self.assertEqual(
        serialized,
        dict(
            zookeepers=('Zoey', 'Zack'),
            animals={
                'tiger': featurespec,
                'elephant': featurespec
            }))
    restored = Zoo.Spec._deserialize(serialized)
    self.assertEqual(zoo_spec, restored)

    # ImmutableDict is used for the field, but dict for the serialization:
    self.assertIsInstance(zoo_spec.animals, immutable_dict.ImmutableDict)
    self.assertIsInstance(serialized['animals'], dict)

  def testSpecComponents(self):

    class Zoo(tensor_struct.Struct):
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


def replace_tensors_with_placeholders(value):

  def repl(x):
    if isinstance(x, ops.Tensor):
      return array_ops.placeholder_with_default(x, shape=None)
    else:
      return x

  return nest.map_structure(repl, value, expand_composites=True)


@contextlib.contextmanager
def temporarily_add_dispatch(op, typ, fn):
  n = len(op._tf_dispatchers)
  dispatch.dispatch_for_types(op, typ)(fn)
  yield
  assert len(op._tf_dispatchers) == n + 1
  del op._tf_dispatchers[-1]


if __name__ == '__main__':
  googletest.main()
