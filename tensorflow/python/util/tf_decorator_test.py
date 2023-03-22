# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for tf_decorator."""

# pylint: disable=unused-import
import functools
import inspect

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


def test_tfdecorator(decorator_name, decorator_doc=None):

  def make_tf_decorator(target):
    return tf_decorator.TFDecorator(decorator_name, target, decorator_doc)

  return make_tf_decorator


def test_decorator_increment_first_int_arg(target):
  """This test decorator skips past `self` as args[0] in the bound case."""

  def wrapper(*args, **kwargs):
    new_args = []
    found = False
    for arg in args:
      if not found and isinstance(arg, int):
        new_args.append(arg + 1)
        found = True
      else:
        new_args.append(arg)
    return target(*new_args, **kwargs)

  return tf_decorator.make_decorator(target, wrapper)


def test_injectable_decorator_square(target):

  def wrapper(x):
    return wrapper.__wrapped__(x)**2

  return tf_decorator.make_decorator(target, wrapper)


def test_injectable_decorator_increment(target):

  def wrapper(x):
    return wrapper.__wrapped__(x) + 1

  return tf_decorator.make_decorator(target, wrapper)


def test_function(x):
  """Test Function Docstring."""
  return x + 1


@test_tfdecorator('decorator 1')
@test_decorator_increment_first_int_arg
@test_tfdecorator('decorator 3', 'decorator 3 documentation')
def test_decorated_function(x):
  """Test Decorated Function Docstring."""
  return x * 2


@test_injectable_decorator_square
@test_injectable_decorator_increment
def test_rewrappable_decorated(x):
  return x * 2


@test_tfdecorator('decorator')
class TestDecoratedClass(object):
  """Test Decorated Class."""

  def __init__(self, two_attr=2):
    self.two_attr = two_attr

  @property
  def two_prop(self):
    return 2

  def two_func(self):
    return 2

  @test_decorator_increment_first_int_arg
  def return_params(self, a, b, c):
    """Return parameters."""
    return [a, b, c]


class TfDecoratorTest(test.TestCase):

  def testInitCapturesTarget(self):
    self.assertIs(test_function,
                  tf_decorator.TFDecorator('', test_function).decorated_target)

  def testInitCapturesDecoratorName(self):
    self.assertEqual(
        'decorator name',
        tf_decorator.TFDecorator('decorator name',
                                 test_function).decorator_name)

  def testInitCapturesDecoratorDoc(self):
    self.assertEqual(
        'decorator doc',
        tf_decorator.TFDecorator('', test_function,
                                 'decorator doc').decorator_doc)

  def testInitCapturesNonNoneArgspec(self):
    argspec = tf_inspect.FullArgSpec(
        args=['a', 'b', 'c'],
        varargs=None,
        varkw=None,
        defaults=(1, 'hello'),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations=None,
    )
    self.assertIs(
        argspec,
        tf_decorator.TFDecorator('', test_function, '',
                                 argspec).decorator_argspec)

  def testInitSetsDecoratorNameToTargetName(self):
    self.assertEqual('test_function',
                     tf_decorator.TFDecorator('', test_function).__name__)

  def testInitSetsDecoratorQualNameToTargetQualName(self):
    if hasattr(tf_decorator.TFDecorator('', test_function), '__qualname__'):
      self.assertEqual('test_function',
                       tf_decorator.TFDecorator('', test_function).__qualname__)

  def testInitSetsDecoratorDocToTargetDoc(self):
    self.assertEqual('Test Function Docstring.',
                     tf_decorator.TFDecorator('', test_function).__doc__)

  def testCallingATFDecoratorCallsTheTarget(self):
    self.assertEqual(124, tf_decorator.TFDecorator('', test_function)(123))

  def testCallingADecoratedFunctionCallsTheTarget(self):
    self.assertEqual((2 + 1) * 2, test_decorated_function(2))

  def testInitializingDecoratedClassWithInitParamsDoesntRaise(self):
    try:
      TestDecoratedClass(2)
    except TypeError:
      self.assertFail()

  def testReadingClassAttributeOnDecoratedClass(self):
    self.assertEqual(2, TestDecoratedClass().two_attr)

  def testCallingClassMethodOnDecoratedClass(self):
    self.assertEqual(2, TestDecoratedClass().two_func())

  def testReadingClassPropertyOnDecoratedClass(self):
    self.assertEqual(2, TestDecoratedClass().two_prop)

  def testNameOnBoundProperty(self):
    self.assertEqual('return_params',
                     TestDecoratedClass().return_params.__name__)

  def testQualNameOnBoundProperty(self):
    if hasattr(TestDecoratedClass().return_params, '__qualname__'):
      self.assertEqual('TestDecoratedClass.return_params',
                       TestDecoratedClass().return_params.__qualname__)

  def testDocstringOnBoundProperty(self):
    self.assertEqual('Return parameters.',
                     TestDecoratedClass().return_params.__doc__)

  def testTarget__get__IsProxied(self):

    class Descr(object):

      def __get__(self, instance, owner):
        return self

    class Foo(object):
      foo = tf_decorator.TFDecorator('Descr', Descr())

    self.assertIsInstance(Foo.foo, Descr)


def test_wrapper(*args, **kwargs):
  return test_function(*args, **kwargs)


class TfMakeDecoratorTest(test.TestCase):

  def testAttachesATFDecoratorAttr(self):
    decorated = tf_decorator.make_decorator(test_function, test_wrapper)
    decorator = getattr(decorated, '_tf_decorator')
    self.assertIsInstance(decorator, tf_decorator.TFDecorator)

  def testAttachesWrappedAttr(self):
    decorated = tf_decorator.make_decorator(test_function, test_wrapper)
    wrapped_attr = getattr(decorated, '__wrapped__')
    self.assertIs(test_function, wrapped_attr)

  def testSetsTFDecoratorNameToDecoratorNameArg(self):
    decorated = tf_decorator.make_decorator(test_function, test_wrapper,
                                            'test decorator name')
    decorator = getattr(decorated, '_tf_decorator')
    self.assertEqual('test decorator name', decorator.decorator_name)

  def testSetsTFDecoratorDocToDecoratorDocArg(self):
    decorated = tf_decorator.make_decorator(
        test_function, test_wrapper, decorator_doc='test decorator doc')
    decorator = getattr(decorated, '_tf_decorator')
    self.assertEqual('test decorator doc', decorator.decorator_doc)

  def testUpdatesDictWithMissingEntries(self):
    test_function.foobar = True
    decorated = tf_decorator.make_decorator(test_function, test_wrapper)
    self.assertTrue(decorated.foobar)
    del test_function.foobar

  def testUpdatesDict_doesNotOverridePresentEntries(self):
    test_function.foobar = True
    test_wrapper.foobar = False
    decorated = tf_decorator.make_decorator(test_function, test_wrapper)
    self.assertFalse(decorated.foobar)
    del test_function.foobar
    del test_wrapper.foobar

  def testSetsTFDecoratorArgSpec(self):
    argspec = tf_inspect.FullArgSpec(
        args=['a', 'b', 'c'],
        varargs='args',
        kwonlyargs={},
        defaults=(1, 'hello'),
        kwonlydefaults=None,
        varkw='kwargs',
        annotations=None)
    decorated = tf_decorator.make_decorator(test_function, test_wrapper, '', '',
                                            argspec)
    decorator = getattr(decorated, '_tf_decorator')
    self.assertEqual(argspec, decorator.decorator_argspec)
    self.assertEqual(
        inspect.signature(decorated),
        inspect.Signature([
            inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                'b', inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1
            ),
            inspect.Parameter(
                'c',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default='hello',
            ),
            inspect.Parameter('args', inspect.Parameter.VAR_POSITIONAL),
            inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD),
        ]),
    )

  def testSetsDecoratorNameToFunctionThatCallsMakeDecoratorIfAbsent(self):

    def test_decorator_name(wrapper):
      return tf_decorator.make_decorator(test_function, wrapper)

    decorated = test_decorator_name(test_wrapper)
    decorator = getattr(decorated, '_tf_decorator')
    self.assertEqual('test_decorator_name', decorator.decorator_name)

  def testCompatibleWithNamelessCallables(self):

    class Callable(object):

      def __call__(self):
        pass

    callable_object = Callable()
    # Smoke test: This should not raise an exception, even though
    # `callable_object` does not have a `__name__` attribute.
    _ = tf_decorator.make_decorator(callable_object, test_wrapper)

    partial = functools.partial(test_function, x=1)
    # Smoke test: This should not raise an exception, even though `partial` does
    # not have `__name__`, `__module__`, and `__doc__` attributes.
    _ = tf_decorator.make_decorator(partial, test_wrapper)


class TfDecoratorRewrapTest(test.TestCase):

  def testRewrapMutatesAffectedFunction(self):

    def new_target(x):
      return x * 3

    self.assertEqual((1 * 2 + 1)**2, test_rewrappable_decorated(1))
    prev_target, _ = tf_decorator.unwrap(test_rewrappable_decorated)
    tf_decorator.rewrap(test_rewrappable_decorated, prev_target, new_target)
    self.assertEqual((1 * 3 + 1)**2, test_rewrappable_decorated(1))

  def testRewrapOfDecoratorFunction(self):

    def new_target(x):
      return x * 3

    prev_target = test_rewrappable_decorated._tf_decorator._decorated_target
    # In this case, only the outer decorator (test_injectable_decorator_square)
    # should be preserved.
    tf_decorator.rewrap(test_rewrappable_decorated, prev_target, new_target)
    self.assertEqual((1 * 3)**2, test_rewrappable_decorated(1))


class TfDecoratorUnwrapTest(test.TestCase):

  def testUnwrapReturnsEmptyArrayForUndecoratedFunction(self):
    decorators, _ = tf_decorator.unwrap(test_function)
    self.assertEqual(0, len(decorators))

  def testUnwrapReturnsUndecoratedFunctionAsTarget(self):
    _, target = tf_decorator.unwrap(test_function)
    self.assertIs(test_function, target)

  def testUnwrapReturnsFinalFunctionAsTarget(self):
    self.assertEqual((4 + 1) * 2, test_decorated_function(4))
    _, target = tf_decorator.unwrap(test_decorated_function)
    self.assertTrue(tf_inspect.isfunction(target))
    self.assertEqual(4 * 2, target(4))

  def testUnwrapReturnsListOfUniqueTFDecorators(self):
    decorators, _ = tf_decorator.unwrap(test_decorated_function)
    self.assertEqual(3, len(decorators))
    self.assertTrue(isinstance(decorators[0], tf_decorator.TFDecorator))
    self.assertTrue(isinstance(decorators[1], tf_decorator.TFDecorator))
    self.assertTrue(isinstance(decorators[2], tf_decorator.TFDecorator))
    self.assertIsNot(decorators[0], decorators[1])
    self.assertIsNot(decorators[1], decorators[2])
    self.assertIsNot(decorators[2], decorators[0])

  def testUnwrapReturnsDecoratorListFromOutermostToInnermost(self):
    decorators, _ = tf_decorator.unwrap(test_decorated_function)
    self.assertEqual('decorator 1', decorators[0].decorator_name)
    self.assertEqual('test_decorator_increment_first_int_arg',
                     decorators[1].decorator_name)
    self.assertEqual('decorator 3', decorators[2].decorator_name)
    self.assertEqual('decorator 3 documentation', decorators[2].decorator_doc)

  def testUnwrapBoundMethods(self):
    test_decorated_class = TestDecoratedClass()
    self.assertEqual([2, 2, 3], test_decorated_class.return_params(1, 2, 3))
    decorators, target = tf_decorator.unwrap(test_decorated_class.return_params)
    self.assertEqual('test_decorator_increment_first_int_arg',
                     decorators[0].decorator_name)
    self.assertEqual([1, 2, 3], target(test_decorated_class, 1, 2, 3))


if __name__ == '__main__':
  test.main()
