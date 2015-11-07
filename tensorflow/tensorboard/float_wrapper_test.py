import tensorflow.python.platform

from tensorflow.python.platform import googletest
from tensorflow.tensorboard import float_wrapper

_INFINITY = float('inf')


class FloatWrapperTest(googletest.TestCase):

  def _assertWrapsAs(self, to_wrap, expected):
    """Asserts that |to_wrap| becomes |expected| when wrapped."""
    actual = float_wrapper.WrapSpecialFloats(to_wrap)
    for a, e in zip(actual, expected):
      self.assertEqual(e, a)

  def testWrapsPrimitives(self):
    self._assertWrapsAs(_INFINITY, 'Infinity')
    self._assertWrapsAs(-_INFINITY, '-Infinity')
    self._assertWrapsAs(float('nan'), 'NaN')

  def testWrapsObjectValues(self):
    self._assertWrapsAs({'x': _INFINITY}, {'x': 'Infinity'})

  def testWrapsObjectKeys(self):
    self._assertWrapsAs({_INFINITY: 'foo'}, {'Infinity': 'foo'})

  def testWrapsInListsAndTuples(self):
    self._assertWrapsAs([_INFINITY], ['Infinity'])
    # map() returns a list even if the argument is a tuple.
    self._assertWrapsAs((_INFINITY,), ['Infinity',])

  def testWrapsRecursively(self):
    self._assertWrapsAs({'x': [_INFINITY]}, {'x': ['Infinity']})


if __name__ == '__main__':
  googletest.main()
