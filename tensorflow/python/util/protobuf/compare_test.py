#!/usr/bin/python2.4

"""Tests for python.util.protobuf.compare."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import re
import textwrap

from tensorflow.python.platform import googletest
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.protobuf import compare_test_pb2
import six

from google.protobuf import text_format


def LargePbs(*args):
  """Converts ASCII string Large PBs to messages."""
  pbs = []
  for arg in args:
    pb = compare_test_pb2.Large()
    text_format.Merge(arg, pb)
    pbs.append(pb)

  return pbs


class Proto2CmpTest(googletest.TestCase):

  def assertGreater(self, a, b):
    """Asserts that Proto2Cmp says a > b."""
    a, b = LargePbs(a, b)
    googletest.TestCase.assertGreater(self, compare.Proto2Cmp(a, b), 0)
    googletest.TestCase.assertLess(self, compare.Proto2Cmp(b, a), 0)

  def assertEquals(self, a, b):
    """Asserts that Proto2Cmp says a == b."""
    a, b = LargePbs(a, b)
    googletest.TestCase.assertEquals(self, compare.Proto2Cmp(a, b), 0)

  def testPrimitives(self):
    googletest.TestCase.assertEqual(self, 0, compare.Proto2Cmp('a', 'a'))
    googletest.TestCase.assertLess(self, 0, compare.Proto2Cmp('b', 'a'))

    pb = compare_test_pb2.Large()
    googletest.TestCase.assertEquals(self, cmp('a', pb), compare.Proto2Cmp('a', pb))
    googletest.TestCase.assertEqual(self, cmp(pb, 'a'), compare.Proto2Cmp(pb, 'a'))

  def testEmpty(self):
    self.assertEquals('', '')

  def testPrimitiveFields(self):
    self.assertGreater('string_: "a"', '')
    self.assertEquals('string_: "a"', 'string_: "a"')
    self.assertGreater('string_: "b"', 'string_: "a"')
    self.assertGreater('string_: "ab"', 'string_: "aa"')

    self.assertGreater('int64_: 0', '')
    self.assertEquals('int64_: 0', 'int64_: 0')
    self.assertGreater('int64_: -1', '')
    self.assertGreater('int64_: 1', 'int64_: 0')
    self.assertGreater('int64_: 0', 'int64_: -1')

    self.assertGreater('float_: 0.0', '')
    self.assertEquals('float_: 0.0', 'float_: 0.0')
    self.assertGreater('float_: -0.1', '')
    self.assertGreater('float_: 3.14', 'float_: 0')
    self.assertGreater('float_: 0', 'float_: -0.1')
    self.assertEquals('float_: -0.1', 'float_: -0.1')

    self.assertGreater('bool_: true', '')
    self.assertGreater('bool_: false', '')
    self.assertGreater('bool_: true', 'bool_: false')
    self.assertEquals('bool_: false', 'bool_: false')
    self.assertEquals('bool_: true', 'bool_: true')

    self.assertGreater('enum_: A', '')
    self.assertGreater('enum_: B', 'enum_: A')
    self.assertGreater('enum_: C', 'enum_: B')
    self.assertEquals('enum_: C', 'enum_: C')

  def testRepeatedPrimitives(self):
    self.assertGreater('int64s: 0', '')
    self.assertEquals('int64s: 0', 'int64s: 0')
    self.assertGreater('int64s: 1', 'int64s: 0')
    self.assertGreater('int64s: 0 int64s: 0', '')
    self.assertGreater('int64s: 0 int64s: 0', 'int64s: 0')
    self.assertGreater('int64s: 1 int64s: 0', 'int64s: 0')
    self.assertGreater('int64s: 0 int64s: 1', 'int64s: 0')
    self.assertGreater('int64s: 1', 'int64s: 0 int64s: 2')
    self.assertGreater('int64s: 2 int64s: 0', 'int64s: 1')
    self.assertEquals('int64s: 0 int64s: 0', 'int64s: 0 int64s: 0')
    self.assertEquals('int64s: 0 int64s: 1', 'int64s: 0 int64s: 1')
    self.assertGreater('int64s: 1 int64s: 0', 'int64s: 0 int64s: 0')
    self.assertGreater('int64s: 1 int64s: 0', 'int64s: 0 int64s: 1')
    self.assertGreater('int64s: 1 int64s: 0', 'int64s: 0 int64s: 2')
    self.assertGreater('int64s: 1 int64s: 1', 'int64s: 1 int64s: 0')
    self.assertGreater('int64s: 1 int64s: 1', 'int64s: 1 int64s: 0 int64s: 2')

  def testMessage(self):
    self.assertGreater('small <>', '')
    self.assertEquals('small <>', 'small <>')
    self.assertGreater('small < strings: "a" >', '')
    self.assertGreater('small < strings: "a" >', 'small <>')
    self.assertEquals('small < strings: "a" >', 'small < strings: "a" >')
    self.assertGreater('small < strings: "b" >', 'small < strings: "a" >')
    self.assertGreater('small < strings: "a" strings: "b" >',
                       'small < strings: "a" >')

    self.assertGreater('string_: "a"', 'small <>')
    self.assertGreater('string_: "a"', 'small < strings: "b" >')
    self.assertGreater('string_: "a"', 'small < strings: "b" strings: "c" >')
    self.assertGreater('string_: "a" small <>', 'small <>')
    self.assertGreater('string_: "a" small <>', 'small < strings: "b" >')
    self.assertEquals('string_: "a" small <>', 'string_: "a" small <>')
    self.assertGreater('string_: "a" small < strings: "a" >',
                       'string_: "a" small <>')
    self.assertEquals('string_: "a" small < strings: "a" >',
                      'string_: "a" small < strings: "a" >')
    self.assertGreater('string_: "a" small < strings: "a" >',
                       'int64_: 1 small < strings: "a" >')
    self.assertGreater('string_: "a" small < strings: "a" >', 'int64_: 1')
    self.assertGreater('string_: "a"', 'int64_: 1 small < strings: "a" >')
    self.assertGreater('string_: "a" int64_: 0 small < strings: "a" >',
                       'int64_: 1 small < strings: "a" >')
    self.assertGreater('string_: "a" int64_: 1 small < strings: "a" >',
                       'string_: "a" int64_: 0 small < strings: "a" >')
    self.assertEquals('string_: "a" int64_: 0 small < strings: "a" >',
                      'string_: "a" int64_: 0 small < strings: "a" >')

  def testNestedMessage(self):
    self.assertGreater('medium <>', '')
    self.assertEquals('medium <>', 'medium <>')
    self.assertGreater('medium < smalls <> >', 'medium <>')
    self.assertEquals('medium < smalls <> >', 'medium < smalls <> >')
    self.assertGreater('medium < smalls <> smalls <> >', 'medium < smalls <> >')
    self.assertEquals('medium < smalls <> smalls <> >',
                      'medium < smalls <> smalls <> >')

    self.assertGreater('medium < int32s: 0 >', 'medium < smalls <> >')

    self.assertGreater('medium < smalls < strings: "a"> >',
                       'medium < smalls <> >')

  def testTagOrder(self):
    """Tests that different fields are ordered by tag number.

    For reference, here are the relevant tag numbers from compare_test.proto:
      optional string string_ = 1;
      optional int64 int64_ = 2;
      optional float float_ = 3;
      optional Small small = 8;
      optional Medium medium = 7;
      optional Small small = 8;
    """
    self.assertGreater('string_: "a"                      ',
                       '             int64_: 1            ')
    self.assertGreater('string_: "a" int64_: 2            ',
                       '             int64_: 1            ')
    self.assertGreater('string_: "b" int64_: 1            ',
                       'string_: "a" int64_: 2            ')
    self.assertEquals( 'string_: "a" int64_: 1            ',
                       'string_: "a" int64_: 1            ')
    self.assertGreater('string_: "a" int64_: 1 float_: 0.0',
                       'string_: "a" int64_: 1            ')
    self.assertEquals( 'string_: "a" int64_: 1 float_: 0.0',
                       'string_: "a" int64_: 1 float_: 0.0')
    self.assertGreater('string_: "a" int64_: 1 float_: 0.1',
                       'string_: "a" int64_: 1 float_: 0.0')
    self.assertGreater('string_: "a" int64_: 2 float_: 0.0',
                       'string_: "a" int64_: 1 float_: 0.1')
    self.assertGreater('string_: "a"                      ',
                       '             int64_: 1 float_: 0.1')
    self.assertGreater('string_: "a"           float_: 0.0',
                       '             int64_: 1            ')
    self.assertGreater('string_: "b"           float_: 0.0',
                       'string_: "a" int64_: 1            ')

    self.assertGreater('string_: "a"',
                       'small < strings: "a" >')
    self.assertGreater('string_: "a" small < strings: "a" >',
                       'small < strings: "b" >')
    self.assertGreater('string_: "a" small < strings: "b" >',
                       'string_: "a" small < strings: "a" >')
    self.assertEquals('string_: "a" small < strings: "a" >',
                      'string_: "a" small < strings: "a" >')

    self.assertGreater('string_: "a" medium <>',
                       'string_: "a" small < strings: "a" >')
    self.assertGreater('string_: "a" medium < smalls <> >',
                       'string_: "a" small < strings: "a" >')
    self.assertGreater('medium <>', 'small < strings: "a" >')
    self.assertGreater('medium <> small <>', 'small < strings: "a" >')
    self.assertGreater('medium < smalls <> >', 'small < strings: "a" >')
    self.assertGreater('medium < smalls < strings: "a" > >',
                       'small < strings: "b" >')


class NormalizeRepeatedFieldsTest(googletest.TestCase):

  def assertNormalizes(self, orig, expected_no_dedupe, expected_dedupe):
    """Checks NormalizeRepeatedFields(orig) against the two expected results."""
    orig, expected_no_dedupe, expected_dedupe = LargePbs(
        orig, expected_no_dedupe, expected_dedupe)

    actual = compare.NormalizeRepeatedFields(copy.deepcopy(orig), dedupe=False)
    self.assertEqual(expected_no_dedupe, actual)

    actual = compare.NormalizeRepeatedFields(copy.deepcopy(orig), dedupe=True)
    self.assertEqual(expected_dedupe, actual)

  def testIgnoreNonRepeatedFields(self):
    orig = """string_: "a" int64_: 1 float_: 0.1 bool_: true enum_: A
              medium: {} small: {}"""
    self.assertNormalizes(orig, orig, orig)

  def testRepeatedPrimitive(self):
    self.assertNormalizes('int64s: 3 int64s: -1 int64s: 2 int64s: -1 int64s: 3',
                          'int64s: -1 int64s: -1 int64s: 2 int64s: 3 int64s: 3',
                          'int64s: -1 int64s: 2 int64s: 3')

  def testRepeatedMessage(self):
    self.assertNormalizes("""medium: { smalls: { strings: "c" }
                                       smalls: { strings: "a" }
                                       smalls: { strings: "b" }
                                       smalls: { strings: "a" }
                                       smalls: { strings: "c" } }
                          """,
                          """medium: { smalls: { strings: "a" }
                                       smalls: { strings: "a" }
                                       smalls: { strings: "b" }
                                       smalls: { strings: "c" }
                                       smalls: { strings: "c" } }
                          """,
                          """medium: { smalls: { strings: "a" }
                                       smalls: { strings: "b" }
                                       smalls: { strings: "c" } }
                          """)

  def testNestedRepeatedGroup(self):
    self.assertNormalizes("""medium {  GroupA { GroupB { strings: "c" }
                                                GroupB { strings: "a" }
                                                GroupB { strings: "b" }
                                                GroupB { strings: "a" }
                                                GroupB { strings: "c" } } }
                          """,
                          """medium {  GroupA { GroupB { strings: "a" }
                                                GroupB { strings: "a" }
                                                GroupB { strings: "b" }
                                                GroupB { strings: "c" }
                                                GroupB { strings: "c" } } }
                          """,
                          """medium {  GroupA { GroupB { strings: "a" }
                                                GroupB { strings: "b" }
                                                GroupB { strings: "c" } } }
                          """)

  def testMapNormalizes(self):
    self.assertNormalizes(
        """with_map: {  value_message: { key: 2, value: { strings: "k2v1",
                                                          strings: "k2v2",
                                                          strings: "k2v1" } },
                        value_message: { key: 1, value: { strings: "k1v2",
                                                          strings: "k1v1" } } }
        """,
        """with_map: {  value_message: { key: 1, value: { strings: "k1v1",
                                                          strings: "k1v2" } },
                        value_message: { key: 2, value: { strings: "k2v1",
                                                          strings: "k2v1",
                                                          strings: "k2v2" } } }
        """,
        """with_map: {  value_message: { key: 1, value: { strings: "k1v1",
                                                          strings: "k1v2" } },
                        value_message: { key: 2, value: { strings: "k2v1",
                                                          strings: "k2v2" } } }
        """)


class NormalizeNumbersTest(googletest.TestCase):
  """Tests for NormalizeNumberFields()."""

  def testNormalizesInts(self):
    pb = compare_test_pb2.Large()
    pb.int64_ = 4
    compare.NormalizeNumberFields(pb)
    self.assertTrue(isinstance(pb.int64_, six.integer_types))

    pb.int64_ = 4
    compare.NormalizeNumberFields(pb)
    self.assertTrue(isinstance(pb.int64_, six.integer_types))

    pb.int64_ = 9999999999999999
    compare.NormalizeNumberFields(pb)
    self.assertTrue(isinstance(pb.int64_, six.integer_types))

  def testNormalizesRepeatedInts(self):
    pb = compare_test_pb2.Large()
    pb.int64s.extend([1, 400, 999999999999999])
    compare.NormalizeNumberFields(pb)
    self.assertTrue(isinstance(pb.int64s[0], six.integer_types))
    self.assertTrue(isinstance(pb.int64s[1], six.integer_types))
    self.assertTrue(isinstance(pb.int64s[2], six.integer_types))

  def testNormalizesFloats(self):
    pb1 = compare_test_pb2.Large()
    pb1.float_ = 1.2314352351231
    pb2 = compare_test_pb2.Large()
    pb2.float_ = 1.231435
    self.assertNotEqual(pb1.float_, pb2.float_)
    compare.NormalizeNumberFields(pb1)
    compare.NormalizeNumberFields(pb2)
    self.assertEqual(pb1.float_, pb2.float_)

  def testNormalizesRepeatedFloats(self):
    pb = compare_test_pb2.Large()
    pb.medium.floats.extend([0.111111111, 0.111111])
    compare.NormalizeNumberFields(pb)
    for value in pb.medium.floats:
      self.assertAlmostEqual(0.111111, value)

  def testNormalizesDoubles(self):
    pb1 = compare_test_pb2.Large()
    pb1.double_ = 1.2314352351231
    pb2 = compare_test_pb2.Large()
    pb2.double_ = 1.2314352
    self.assertNotEqual(pb1.double_, pb2.double_)
    compare.NormalizeNumberFields(pb1)
    compare.NormalizeNumberFields(pb2)
    self.assertEqual(pb1.double_, pb2.double_)

  def testNormalizesMaps(self):
    pb = compare_test_pb2.WithMap()
    pb.value_message[4].strings.extend(['a', 'b', 'c'])
    pb.value_string['d'] = 'e'
    compare.NormalizeNumberFields(pb)


class AssertTest(googletest.TestCase):
  """Tests both assertProto2Equal() and assertProto2SameElements()."""
  def assertProto2Equal(self, a, b, **kwargs):
    if isinstance(a, six.string_types) and isinstance(b, six.string_types):
      a, b = LargePbs(a, b)
    compare.assertProto2Equal(self, a, b, **kwargs)

  def assertProto2SameElements(self, a, b, **kwargs):
    if isinstance(a, six.string_types) and isinstance(b, six.string_types):
      a, b = LargePbs(a, b)
    compare.assertProto2SameElements(self, a, b, **kwargs)

  def assertAll(self, a, **kwargs):
    """Checks that all possible asserts pass."""
    self.assertProto2Equal(a, a, **kwargs)
    self.assertProto2SameElements(a, a, number_matters=False, **kwargs)
    self.assertProto2SameElements(a, a, number_matters=True, **kwargs)

  def assertSameNotEqual(self, a, b):
    """Checks that assertProto2SameElements() passes with number_matters=False
    and number_matters=True but not assertProto2Equal().
    """
    self.assertProto2SameElements(a, b, number_matters=False)
    self.assertProto2SameElements(a, b, number_matters=True)
    self.assertRaises(AssertionError, self.assertProto2Equal, a, b)

  def assertSameExceptNumber(self, a, b):
    """Checks that assertProto2SameElements() passes with number_matters=False
    but not number_matters=True or assertProto2Equal().
    """
    self.assertProto2SameElements(a, b, number_matters=False)
    self.assertRaises(AssertionError, self.assertProto2SameElements, a, b,
                      number_matters=True)
    self.assertRaises(AssertionError, self.assertProto2Equal, a, b)

  def assertNone(self, a, b, message, **kwargs):
    """Checks that all possible asserts fail with the given message."""
    message = re.escape(textwrap.dedent(message))
    self.assertRaisesRegexp(AssertionError, message,
                            self.assertProto2SameElements, a, b,
                            number_matters=False, **kwargs)
    self.assertRaisesRegexp(AssertionError, message,
                            self.assertProto2SameElements, a, b,
                            number_matters=True, **kwargs)
    self.assertRaisesRegexp(AssertionError, message,
                            self.assertProto2Equal, a, b, **kwargs)

  def testCheckInitialized(self):
    # neither is initialized
    a = compare_test_pb2.Labeled()
    a.optional = 1
    self.assertNone(a, a, 'Initialization errors: ', check_initialized=True)
    self.assertAll(a, check_initialized=False)

    # a is initialized, b isn't
    b = copy.deepcopy(a)
    a.required = 2
    self.assertNone(a, b, 'Initialization errors: ', check_initialized=True)
    self.assertNone(a, b,
                    """
                    - required: 2
                      optional: 1
                    """,
                    check_initialized=False)

    # both are initialized
    a = compare_test_pb2.Labeled()
    a.required = 2
    self.assertAll(a, check_initialized=True)
    self.assertAll(a, check_initialized=False)

    b = copy.deepcopy(a)
    b.required = 3
    message = """
              - required: 2
              ?           ^
              + required: 3
              ?           ^
              """
    self.assertNone(a, b, message, check_initialized=True)
    self.assertNone(a, b, message, check_initialized=False)

  def testAssertEqualWithStringArg(self):
    pb = compare_test_pb2.Large()
    pb.string_ = 'abc'
    pb.float_ = 1.234
    compare.assertProto2Equal(
        self,
        """
          string_: 'abc'
          float_: 1.234
        """,
        pb)

  def testAssertSameElementsWithStringArg(self):
    pb = compare_test_pb2.Large()
    pb.string_ = 'abc'
    pb.float_ = 1.234
    pb.int64s.extend([7, 3, 5])
    compare.assertProto2SameElements(
        self,
        """
          string_: 'abc'
          float_: 1.234
          int64s: 3
          int64s: 7
          int64s: 5
        """,
        pb)

  def testProto2ContainsString(self):
    pb = compare_test_pb2.Large()
    pb.string_ = 'abc'
    pb.float_ = 1.234
    pb.small.strings.append('xyz')
    compare.assertProto2Contains(
        self,
        """
          small {
            strings: "xyz"
          }
        """,
        pb)

  def testProto2ContainsProto(self):
    pb = compare_test_pb2.Large()
    pb.string_ = 'abc'
    pb.float_ = 1.234
    pb.small.strings.append('xyz')
    pb2 = compare_test_pb2.Large()
    pb2.small.strings.append('xyz')
    compare.assertProto2Contains(
        self, pb2, pb)

  def testNormalizesNumbers(self):
    pb1 = compare_test_pb2.Large()
    pb1.int64_ = 4
    pb2 = compare_test_pb2.Large()
    pb2.int64_ = 4
    compare.assertProto2Equal(self, pb1, pb2)

  def testNormalizesFloat(self):
    pb1 = compare_test_pb2.Large()
    pb1.double_ = 4.0
    pb2 = compare_test_pb2.Large()
    pb2.double_ = 4
    compare.assertProto2Equal(self, pb1, pb2, normalize_numbers=True)

    pb1 = compare_test_pb2.Medium()
    pb1.floats.extend([4.0, 6.0])
    pb2 = compare_test_pb2.Medium()
    pb2.floats.extend([6, 4])
    compare.assertProto2SameElements(self, pb1, pb2, normalize_numbers=True)

  def testPrimitives(self):
    self.assertAll('string_: "x"')
    self.assertNone('string_: "x"',
                    'string_: "y"',
                    """
                    - string_: "x"
                    ?           ^
                    + string_: "y"
                    ?           ^
                    """)

  def testRepeatedPrimitives(self):
    self.assertAll('int64s: 0 int64s: 1')

    self.assertSameNotEqual('int64s: 0 int64s: 1', 'int64s: 1 int64s: 0')
    self.assertSameNotEqual('int64s: 0 int64s: 1 int64s: 2',
                            'int64s: 2 int64s: 1 int64s: 0')

    self.assertSameExceptNumber('int64s: 0', 'int64s: 0 int64s: 0')
    self.assertSameExceptNumber('int64s: 0 int64s: 1',
                                'int64s: 1 int64s: 0 int64s: 1')

    self.assertNone('int64s: 0',
                    'int64s: 0 int64s: 2',
                    """
                      int64s: 0
                    + int64s: 2
                    """)
    self.assertNone('int64s: 0 int64s: 1',
                    'int64s: 0 int64s: 2',
                    """
                      int64s: 0
                    - int64s: 1
                    ?         ^
                    + int64s: 2
                    ?         ^
                    """)

  def testMessage(self):
    self.assertAll('medium: {}')
    self.assertAll('medium: { smalls: {} }')
    self.assertAll('medium: { int32s: 1 smalls: {} }')
    self.assertAll('medium: { smalls: { strings: "x" } }')
    self.assertAll('medium: { smalls: { strings: "x" } } small: { strings: "y" }')

    self.assertSameNotEqual(
        'medium: { smalls: { strings: "x" strings: "y" } }',
        'medium: { smalls: { strings: "y" strings: "x" } }')
    self.assertSameNotEqual(
        'medium: { smalls: { strings: "x" } smalls: { strings: "y" } }',
        'medium: { smalls: { strings: "y" } smalls: { strings: "x" } }')

    self.assertSameExceptNumber(
        'medium: { smalls: { strings: "x" strings: "y" strings: "x" } }',
        'medium: { smalls: { strings: "y" strings: "x" } }')
    self.assertSameExceptNumber(
        'medium: { smalls: { strings: "x" } int32s: 0 }',
        'medium: { int32s: 0 smalls: { strings: "x" } int32s: 0 }')

    self.assertNone('medium: {}',
                    'medium: { smalls: { strings: "x" } }',
                    """
                      medium {
                    +   smalls {
                    +     strings: "x"
                    +   }
                      }
                    """)
    self.assertNone('medium: { smalls: { strings: "x" } }',
                    'medium: { smalls: {} }',
                    """
                      medium {
                        smalls {
                    -     strings: "x"
                        }
                      }
                    """)
    self.assertNone('medium: { int32s: 0 }',
                    'medium: { int32s: 1 }',
                    """
                      medium {
                    -   int32s: 0
                    ?           ^
                    +   int32s: 1
                    ?           ^
                      }
                    """)

  def testMsgPassdown(self):
    self.assertRaisesRegexp(AssertionError, 'test message passed down',
                            self.assertProto2Equal,
                            'medium: {}',
                            'medium: { smalls: { strings: "x" } }',
                            msg='test message passed down')

  def testRepeatedMessage(self):
    self.assertAll('medium: { smalls: {} smalls: {} }')
    self.assertAll('medium: { smalls: { strings: "x" } } medium: {}')
    self.assertAll('medium: { smalls: { strings: "x" } } medium: { int32s: 0 }')
    self.assertAll('medium: { smalls: {} smalls: { strings: "x" } } small: {}')

    self.assertSameNotEqual('medium: { smalls: { strings: "x" } smalls: {} }',
                            'medium: { smalls: {} smalls: { strings: "x" } }')

    self.assertSameExceptNumber('medium: { smalls: {} }',
                                'medium: { smalls: {} smalls: {} }')
    self.assertSameExceptNumber('medium: { smalls: {} smalls: {} } medium: {}',
                                'medium: {} medium: {} medium: { smalls: {} }')
    self.assertSameExceptNumber(
        'medium: { smalls: { strings: "x" } smalls: {} }',
        'medium: { smalls: {} smalls: { strings: "x" } smalls: {} }')

    self.assertNone('medium: {}',
                    'medium: {} medium { smalls: {} }',
                    """
                      medium {
                    +   smalls {
                    +   }
                      }
                    """)
    self.assertNone('medium: { smalls: {} smalls: { strings: "x" } }',
                    'medium: { smalls: {} smalls: { strings: "y" } }',
                    """
                      medium {
                        smalls {
                        }
                        smalls {
                    -     strings: "x"
                    ?               ^
                    +     strings: "y"
                    ?               ^
                        }
                      }
                    """)


class MixinTests(compare.Proto2Assertions, googletest.TestCase):

  def testAssertEqualWithStringArg(self):
    pb = compare_test_pb2.Large()
    pb.string_ = 'abc'
    pb.float_ = 1.234
    self.assertProto2Equal(
        """
          string_: 'abc'
          float_: 1.234
        """,
        pb)

  def testAssertSameElements(self):
    a = compare_test_pb2.Large()
    a.string_ = 'abc'
    a.float_ = 1.234
    a.int64s[:] = [4, 3, 2]
    b = compare_test_pb2.Large()
    b.CopyFrom(a)
    b.int64s[:] = [2, 4, 3]
    self.assertProto2SameElements(a, b)


if __name__ == '__main__':
  googletest.main()
