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
"""Tests for tensorflow.python.framework.python_api_dispatcher."""

import collections
import numpy as np

from tensorflow.python.framework import _pywrap_python_api_dispatcher as dispatch
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest

# PyTypeChecker::MatchType enum values.
NO_MATCH = dispatch.MatchType.NO_MATCH
MATCH = dispatch.MatchType.MATCH
MATCH_DISPATCHABLE = dispatch.MatchType.MATCH_DISPATCHABLE


@test_util.run_all_in_graph_and_eager_modes
class PythonTypeCheckerTest(test_util.TensorFlowTestCase):

  def testInstanceChecker(self):
    t = constant_op.constant([1, 2, 3])
    rt = ragged_factory_ops.constant([[1, 2], [3, 4, 5]])

    with self.subTest('int checker'):
      int_checker = dispatch.MakeInstanceChecker(int)
      self.assertEqual(int_checker.Check(3), MATCH)
      self.assertEqual(int_checker.Check(3.0), NO_MATCH)
      self.assertEqual(int_checker.Check(t), NO_MATCH)
      self.assertEqual(int_checker.cost(), 1)
      self.assertEqual(repr(int_checker), '<PyTypeChecker int>')

    with self.subTest('tensor checker'):
      tensor_checker = dispatch.MakeInstanceChecker(ops.Tensor)
      self.assertEqual(tensor_checker.Check(t), MATCH)
      self.assertEqual(tensor_checker.Check(3), NO_MATCH)
      self.assertEqual(tensor_checker.Check(3.0), NO_MATCH)
      self.assertEqual(tensor_checker.cost(), 1)
      self.assertEqual(repr(tensor_checker), '<PyTypeChecker Tensor>')

    with self.subTest('ragged checker'):
      ragged_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
      self.assertEqual(ragged_checker.Check(rt), MATCH_DISPATCHABLE)
      self.assertEqual(ragged_checker.Check(3), NO_MATCH)
      self.assertEqual(ragged_checker.Check(t), NO_MATCH)
      self.assertEqual(ragged_checker.cost(), 1)
      self.assertEqual(repr(ragged_checker), '<PyTypeChecker RaggedTensor>')

    with self.subTest('int or float checker'):
      int_checker = dispatch.MakeInstanceChecker(int, float)
      self.assertEqual(int_checker.Check(3), MATCH)
      self.assertEqual(int_checker.Check(3.0), MATCH)
      self.assertEqual(int_checker.Check(t), NO_MATCH)
      self.assertEqual(int_checker.cost(), 2)
      self.assertEqual(repr(int_checker), '<PyTypeChecker int, float>')

    with self.subTest('subclasses'):

      class A(object):
        pass

      class B(A):
        pass

      class C(object):
        pass

      class D(C, B):
        pass

      checker = dispatch.MakeInstanceChecker(A)
      self.assertEqual(checker.Check(A()), MATCH)
      self.assertEqual(checker.Check(B()), MATCH)
      self.assertEqual(checker.Check(C()), NO_MATCH)
      self.assertEqual(checker.Check(D()), MATCH)

  def testInstanceCheckerCache(self):
    checker = dispatch.MakeInstanceChecker(tuple)
    MyTuple = collections.namedtuple('MyTuple', ['a', 'b'])  # Subclass of tuple

    self.assertEqual(checker.cache_size(), 0)
    self.assertEqual(checker.Check(5), NO_MATCH)
    self.assertEqual(checker.cache_size(), 1)  # cache miss
    self.assertEqual(checker.Check(12), NO_MATCH)
    self.assertEqual(checker.cache_size(), 1)  # cache hit
    self.assertEqual(checker.Check(1.3), NO_MATCH)
    self.assertEqual(checker.cache_size(), 2)  # cache miss
    self.assertEqual(checker.Check([1]), NO_MATCH)
    self.assertEqual(checker.cache_size(), 3)  # cache miss
    self.assertEqual(checker.Check((1,)), MATCH)
    self.assertEqual(checker.cache_size(), 4)  # cache miss
    self.assertEqual(checker.Check((1, 2, 3)), MATCH)
    self.assertEqual(checker.cache_size(), 4)  # cache hit
    self.assertEqual(checker.Check(MyTuple(1, 2)), MATCH)
    self.assertEqual(checker.cache_size(), 5)  # cache miss
    self.assertEqual(checker.Check(MyTuple(3, 4)), MATCH)
    self.assertEqual(checker.cache_size(), 5)  # cache miss
    self.assertEqual(checker.Check(()), MATCH)
    self.assertEqual(checker.cache_size(), 5)  # cache hit

  def testUnionChecker(self):
    int_checker = dispatch.MakeInstanceChecker(int)
    float_checker = dispatch.MakeInstanceChecker(float)
    str_checker = dispatch.MakeInstanceChecker(str)
    none_checker = dispatch.MakeInstanceChecker(type(None))
    tensor_checker = dispatch.MakeInstanceChecker(ops.Tensor)
    ragged_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)

    t = constant_op.constant([1, 2, 3])
    rt = ragged_factory_ops.constant([[1, 2], [3, 4, 5]])

    with self.subTest('Union[int, float, str]'):
      checker = dispatch.MakeUnionChecker(
          [int_checker, float_checker, str_checker])
      self.assertEqual(checker.Check(3), MATCH)
      self.assertEqual(checker.Check(3.0), MATCH)
      self.assertEqual(checker.Check('x'), MATCH)
      self.assertEqual(checker.Check('x'), MATCH)
      self.assertEqual(checker.Check(None), NO_MATCH)
      self.assertEqual(checker.Check(t), NO_MATCH)
      self.assertEqual(checker.cost(), 4)
      self.assertEqual(repr(checker), '<PyTypeChecker Union[int, float, str]>')

    with self.subTest('Optional[int] (aka Union[int, None])'):
      checker = dispatch.MakeUnionChecker([int_checker, none_checker])
      self.assertEqual(checker.Check(3), MATCH)
      self.assertEqual(checker.Check(3.0), NO_MATCH)
      self.assertEqual(checker.Check(None), MATCH)
      self.assertEqual(checker.Check(t), NO_MATCH)
      self.assertEqual(checker.cost(), 3)
      self.assertEqual(repr(checker), '<PyTypeChecker Union[int, NoneType]>')

    with self.subTest('Union[Tensor, RaggedTensor]'):
      checker = dispatch.MakeUnionChecker([tensor_checker, ragged_checker])
      self.assertEqual(checker.Check(3), NO_MATCH)
      self.assertEqual(checker.Check(3.0), NO_MATCH)
      self.assertEqual(checker.Check(None), NO_MATCH)
      self.assertEqual(checker.Check(t), MATCH)
      self.assertEqual(checker.Check(rt), MATCH_DISPATCHABLE)
      self.assertEqual(checker.cost(), 3)
      self.assertEqual(
          repr(checker), '<PyTypeChecker Union[Tensor, RaggedTensor]>')

  def testListChecker(self):
    int_checker = dispatch.MakeInstanceChecker(int)
    tensor_checker = dispatch.MakeInstanceChecker(ops.Tensor)
    ragged_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    np_int_checker = dispatch.MakeInstanceChecker(np.integer)

    t = constant_op.constant([1, 2, 3])
    rt = ragged_factory_ops.constant([[1, 2], [3, 4, 5]])
    a = [1, 2, 3]
    b = ['a', 2, t]
    c = [t, t * 2, t - 2]
    d = [t, rt]
    e = []
    f = (1, 2, 3)
    g = (rt,)
    h = {1: 2, 3: 4}
    i = np.array([1, 2, 3])

    with self.subTest('List[int]'):
      checker = dispatch.MakeListChecker(int_checker)
      self.assertEqual(checker.Check(a), MATCH)
      self.assertEqual(checker.Check(b), NO_MATCH)
      self.assertEqual(checker.Check(c), NO_MATCH)
      self.assertEqual(checker.Check(d), NO_MATCH)
      self.assertEqual(checker.Check(e), MATCH)
      self.assertEqual(checker.Check(f), MATCH)
      self.assertEqual(checker.Check(iter(a)), NO_MATCH)
      self.assertEqual(checker.Check(iter(b)), NO_MATCH)
      self.assertEqual(checker.Check(reversed(e)), NO_MATCH)
      self.assertEqual(checker.Check(h), NO_MATCH)
      self.assertEqual(checker.Check(i), NO_MATCH)
      self.assertEqual(checker.cost(), 10)
      self.assertEqual(repr(checker), '<PyTypeChecker List[int]>')

    with self.subTest('List[Tensor]'):
      checker = dispatch.MakeListChecker(tensor_checker)
      self.assertEqual(checker.Check(a), NO_MATCH)
      self.assertEqual(checker.Check(b), NO_MATCH)
      self.assertEqual(checker.Check(c), MATCH)
      self.assertEqual(checker.Check(d), NO_MATCH)
      self.assertEqual(checker.Check(e), MATCH)
      self.assertEqual(checker.cost(), 10)
      self.assertEqual(repr(checker), '<PyTypeChecker List[Tensor]>')

    with self.subTest('List[Union[Tensor, RaggedTensor]]'):
      checker = dispatch.MakeListChecker(
          dispatch.MakeUnionChecker([tensor_checker, ragged_checker]))
      self.assertEqual(checker.Check(a), NO_MATCH)
      self.assertEqual(checker.Check(b), NO_MATCH)
      self.assertEqual(checker.Check(c), MATCH)
      self.assertEqual(checker.Check(d), MATCH_DISPATCHABLE)
      self.assertEqual(checker.Check(e), MATCH)
      self.assertEqual(checker.Check(f), NO_MATCH)
      self.assertEqual(checker.Check(g), MATCH_DISPATCHABLE)
      self.assertEqual(checker.cost(), 30)
      self.assertEqual(
          repr(checker), '<PyTypeChecker List[Union[Tensor, RaggedTensor]]>')

    with self.subTest('List[Union[int, np.integer]]'):
      # Note: np.integer is a subtype of int in *some* Python versions.
      checker = dispatch.MakeListChecker(
          dispatch.MakeUnionChecker([int_checker, np_int_checker]))
      self.assertEqual(checker.Check(a), MATCH)
      self.assertEqual(checker.Check(np.array(a)), NO_MATCH)
      self.assertEqual(checker.Check(np.array(a) * 1.5), NO_MATCH)

  def testRegisterDispatchableType(self):

    @dispatch.register_dispatchable_type
    class A(object):
      pass

    checker = dispatch.MakeInstanceChecker(A)
    self.assertEqual(checker.Check(A()), MATCH_DISPATCHABLE)

  def testRegisterDispatchableTypeError(self):

    with self.assertRaisesRegex(ValueError, 'Expected a type object'):
      dispatch.register_dispatchable_type(3)
    with self.assertRaisesRegex(ValueError,
                                'Type .* has already been registered'):
      dispatch.register_dispatchable_type(ragged_tensor.RaggedTensor)


@test_util.run_all_in_graph_and_eager_modes
class PythonSignatureCheckerTest(test_util.TensorFlowTestCase):

  def check_signatures(self, checker, canon_expected_pairs):
    for (canon_args, expected) in canon_expected_pairs:
      with self.subTest(f'{canon_args} -> {expected}'):
        self.assertEqual(checker.CheckCanonicalizedArgs(canon_args), expected)

  def testSimpleSignature(self):
    int_checker = dispatch.MakeInstanceChecker(int)
    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    checker = dispatch.PySignatureChecker([(0, int_checker), (2, rt_checker)])
    rt = ragged_factory_ops.constant([[1, 2], [3]])

    self.check_signatures(checker, [
        ((1, 2, rt), True),
        ((1, 2, 3), False),
        ((1, 2), False), ((), False),
        ((5, 'x', rt, None), True),
        (([5], 'x', rt, None), False),
        ((5, 'x', [rt], None), False),
    ])  # pyformat: disable

    self.assertEqual(
        repr(checker), '<PySignatureChecker args[0]:int, args[2]:RaggedTensor>')

  def testUnion(self):
    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    tensor_checker = dispatch.MakeInstanceChecker(ops.Tensor)
    rt_or_tensor = dispatch.MakeUnionChecker([rt_checker, tensor_checker])
    checker = dispatch.PySignatureChecker([(0, rt_or_tensor),
                                           (1, rt_or_tensor)])

    t = constant_op.constant([[1, 2], [3, 4]])
    rt = ragged_factory_ops.constant([[1, 2], [3]])

    self.check_signatures(checker, [
        ((t, t), False),
        ((t, rt), True),
        ((rt, t), True),
        ((rt, rt), True),
        ((rt, [rt]), False),
        ((rt, rt, 1, 2, None), True),
    ])  # pyformat: disable

    self.assertEqual(
        repr(checker),
        '<PySignatureChecker args[0]:Union[RaggedTensor, Tensor], '
        'args[1]:Union[RaggedTensor, Tensor]>')

  def testList(self):
    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    rt_list_checker = dispatch.MakeListChecker(rt_checker)
    checker = dispatch.PySignatureChecker([(0, rt_list_checker)])

    rt = ragged_factory_ops.constant([[1, 2], [3]])

    self.check_signatures(checker, [
        (([rt],), True),
        (([],), False),
        ((rt,), False),
        (([rt, rt+3, rt*2],), True),
        (([rt, rt.values, rt*2],), False),
    ])  # pyformat: disable

    self.assertEqual(
        repr(checker), '<PySignatureChecker args[0]:List[RaggedTensor]>')

  def testSortByCost(self):
    a = dispatch.MakeInstanceChecker(int)
    b = dispatch.MakeInstanceChecker(float)
    c = dispatch.MakeUnionChecker([a, b])
    d = dispatch.MakeListChecker(a)
    e = dispatch.MakeListChecker(c)
    checker = dispatch.PySignatureChecker([(0, e), (1, c), (2, d), (3, a)])

    # Note: `repr(checker)` lists the args in the order they will be checked.
    self.assertEqual(
        repr(checker), '<PySignatureChecker '
        'args[3]:int, '                     # a: cost=1
        'args[1]:Union[int, float], '       # c: cost=3
        'args[2]:List[int], '               # d: cost=10
        'args[0]:List[Union[int, float]]>'  # e: cost=30
        )  # pyformat: disable


@test_util.run_all_in_graph_and_eager_modes
class PythonAPIDispatcherTest(test_util.TensorFlowTestCase):

  def testBasicDispatch(self):
    dispatcher = dispatch.PythonAPIDispatcher('tf.foo', ['x', 'y', 'name'],
                                              (None,))

    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    f1 = lambda x, y, name=None: 'f1'
    dispatcher.Register(dispatch.PySignatureChecker([(0, rt_checker)]), f1)

    rt = ragged_factory_ops.constant([[1, 2], [3]])
    self.assertEqual(dispatcher.Dispatch((rt, 5), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((rt, 5, 'my_name'), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((), {'x': rt, 'y': 5}), 'f1')
    self.assertEqual(
        dispatcher.Dispatch((), {
            'x': rt,
            'y': 5,
            'name': 'x'
        }), 'f1')
    self.assertEqual(dispatcher.Dispatch(('foo', rt), None), NotImplemented)
    self.assertEqual(dispatcher.Dispatch(('foo', 'bar'), None), NotImplemented)
    self.assertEqual(
        dispatcher.Dispatch(('foo', 'bar', 'baz'), None), NotImplemented)

  def testMultipleDispatchers(self):
    dispatcher = dispatch.PythonAPIDispatcher('tf.foo', ['x', 'y', 'name'],
                                              (None,))

    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    rt_x_checker = dispatch.PySignatureChecker([(0, rt_checker)])
    rt_y_checker = dispatch.PySignatureChecker([(1, rt_checker)])

    f1 = lambda x, y, name=None: 'f1'
    f2 = lambda x, y, name=None: 'f2'

    rt = ragged_factory_ops.constant([[1, 2], [3]])

    dispatcher.Register(rt_x_checker, f1)
    dispatcher.Register(rt_y_checker, f2)

    self.assertEqual(dispatcher.Dispatch((rt, 5), None), 'f1')
    self.assertEqual(dispatcher.Dispatch(('foo', rt), None), 'f2')
    self.assertEqual(dispatcher.Dispatch(('foo',), {'y': rt}), 'f2')
    self.assertEqual(dispatcher.Dispatch(('foo', 'bar'), None), NotImplemented)
    with self.assertRaisesRegex(
        ValueError, 'Multiple dispatch targets .*'
        r'match the arguments to tf\.foo'):
      dispatcher.Dispatch((rt, rt), None)

  def testListAndUnionDispatch(self):
    dispatcher = dispatch.PythonAPIDispatcher('tf.foo', ['x', 'ys', 'name'],
                                              (None,))

    rt_checker = dispatch.MakeInstanceChecker(ragged_tensor.RaggedTensor)
    tensor_checker = dispatch.MakeInstanceChecker(ops.Tensor)
    rt_or_t = dispatch.MakeUnionChecker([rt_checker, tensor_checker])
    list_of_rt_or_t = dispatch.MakeListChecker(rt_or_t)
    f1 = lambda x, ys, name=None: 'f1'
    dispatcher.Register(
        dispatch.PySignatureChecker([(0, rt_or_t), (1, list_of_rt_or_t)]), f1)

    rt = ragged_factory_ops.constant([[1, 2], [3]])
    t = constant_op.constant(5)
    self.assertEqual(dispatcher.Dispatch((rt, [t]), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((rt, [rt]), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((t, [rt]), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((rt, []), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((t, [t, t, rt, t]), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((rt, [t], 'my_name'), None), 'f1')
    self.assertEqual(dispatcher.Dispatch((), {'x': rt, 'ys': [t]}), 'f1')
    self.assertEqual(
        dispatcher.Dispatch((), {
            'x': rt,
            'ys': [t],
            'name': 'x'
        }), 'f1')
    self.assertEqual(dispatcher.Dispatch((t, [t]), None), NotImplemented)
    self.assertEqual(dispatcher.Dispatch((t, []), None), NotImplemented)
    self.assertEqual(dispatcher.Dispatch(('foo', [rt]), None), NotImplemented)
    self.assertEqual(dispatcher.Dispatch(('foo', 'bar'), None), NotImplemented)
    self.assertEqual(
        dispatcher.Dispatch(('foo', 'bar', 'baz'), None), NotImplemented)


if __name__ == '__main__':
  googletest.main()
