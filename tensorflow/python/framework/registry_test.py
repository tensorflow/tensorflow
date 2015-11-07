"""Tests for tensorflow.ops.registry."""

from tensorflow.python.framework import registry
from tensorflow.python.platform import googletest


class RegistryTest(googletest.TestCase):

  class Foo(object):
    pass

  def testRegisterClass(self):
    myreg = registry.Registry('testfoo')
    with self.assertRaises(LookupError):
      myreg.lookup('Foo')
    myreg.register(RegistryTest.Foo, 'Foo')
    assert myreg.lookup('Foo') == RegistryTest.Foo

  def testRegisterFunction(self):
    myreg = registry.Registry('testbar')
    with self.assertRaises(LookupError):
      myreg.lookup('Bar')
    myreg.register(bar, 'Bar')
    assert myreg.lookup('Bar') == bar

  def testDuplicate(self):
    myreg = registry.Registry('testbar')
    myreg.register(bar, 'Bar')
    with self.assertRaises(KeyError):
      myreg.register(bar, 'Bar')


def bar():
  pass


if __name__ == '__main__':
  googletest.main()
