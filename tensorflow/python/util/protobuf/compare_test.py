# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for python.util.protobuf.compare."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import re
import textwrap

import six

from tensorflow.python.platform import googletest
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.protobuf import compare_test_pb2

from google.protobuf import text_format


def LargePbs(*args):
  """Converts ASCII string Large PBs to messages."""
  pbs = []
  for arg in args:
    pb = compare_test_pb2.Large()
    text_format.Merge(arg, pb)
    pbs.append(pb)

  return pbs


class ProtoEqTest(googletest.TestCase):

  def assertNotEquals(self, a, b):
    """Asserts that ProtoEq says a != b."""
    a, b = LargePbs(a, b)
    googletest.TestCase.assertEquals(self, compare.ProtoEq(a, b), False)

  def assertEquals(self, a, b):
    """Asserts that ProtoEq says a == b."""
    a, b = LargePbs(a, b)
    googletest.TestCase.assertEquals(self, compare.ProtoEq(a, b), True)

  def testPrimitives(self):
    googletest.TestCase.assertEqual(self, True, compare.ProtoEq('a', 'a'))
    googletest.TestCase.assertEqual(self, False, compare.ProtoEq('b', 'a'))

  def testEmpty(self):
    self.assertEquals('', '')

  def testPrimitiveFields(self):
    self.assertNotEquals('string_: "a"', '')
    self.assertEquals('string_: "a"', 'string_: "a"')
    self.assertNotEquals('string_: "b"', 'string_: "a"')
    self.assertNotEquals('string_: "ab"', 'string_: "aa"')

    self.assertNotEquals('int64_: 0', '')
    self.assertEquals('int64_: 0', 'int64_: 0')
    self.assertNotEquals('int64_: -1', '')
    self.assertNotEquals('int64_: 1', 'int64_: 0')
    self.assertNotEquals('int64_: 0', 'int64_: -1')

    self.assertNotEquals('float_: 0.0', '')
    self.assertEquals('float_: 0.0', 'float_: 0.0')
    self.assertNotEquals('float_: -0.1', '')
    self.assertNotEquals('float_: 3.14', 'float_: 0')
    self.assertNotEquals('float_: 0', 'float_: -0.1')
    self.assertEquals('float_: -0.1', 'float_: -0.1')

    self.assertNotEquals('bool_: true', '')
    self.assertNotEquals('bool_: false', '')
    self.assertNotEquals('bool_: true', 'bool_: false')
    self.assertEquals('bool_: false', 'bool_: false')
    self.assertEquals('bool_: true', 'bool_: true')

    self.assertNotEquals('enum_: A', '')
    self.assertNotEquals('enum_: B', 'enum_: A')
    self.assertNotEquals('enum_: C', 'enum_: B')
    self.assertEquals('enum_: C', 'enum_: C')

  def testRepeatedPrimitives(self):
    self.assertNotEquals('int64s: 0', '')
    self.assertEquals('int64s: 0', 'int64s: 0')
    self.assertNotEquals('int64s: 1', 'int64s: 0')
    self.assertNotEquals('int64s: 0 int64s: 0', '')
    self.assertNotEquals('int64s: 0 int64s: 0', 'int64s: 0')
    self.assertNotEquals('int64s: 1 int64s: 0', 'int64s: 0')
    self.assertNotEquals('int64s: 0 int64s: 1', 'int64s: 0')
    self.assertNotEquals('int64s: 1', 'int64s: 0 int64s: 2')
    self.assertNotEquals('int64s: 2 int64s: 0', 'int64s: 1')
    self.assertEquals('int64s: 0 int64s: 0', 'int64s: 0 int64s: 0')
    self.assertEquals('int64s: 0 int64s: 1', 'int64s: 0 int64s: 1')
    self.assertNotEquals('int64s: 1 int64s: 0', 'int64s: 0 int64s: 0')
    self.assertNotEquals('int64s: 1 int64s: 0', 'int64s: 0 int64s: 1')
    self.assertNotEquals('int64s: 1 int64s: 0', 'int64s: 0 int64s: 2')
    self.assertNotEquals('int64s: 1 int64s: 1', 'int64s: 1 int64s: 0')
    self.assertNotEquals('int64s: 1 int64s: 1', 'int64s: 1 int64s: 0 int64s: 2')

  def testMessage(self):
    self.assertNotEquals('small <>', '')
    self.assertEquals('small <>', 'small <>')
    self.assertNotEquals('small < strings: "a" >', '')
    self.assertNotEquals('small < strings: "a" >', 'small <>')
    self.assertEquals('small < strings: "a" >', 'small < strings: "a" >')
    self.assertNotEquals('small < strings: "b" >', 'small < strings: "a" >')
    self.assertNotEquals('small < strings: "a" strings: "b" >',
                         'small < strings: "a" >')

    self.assertNotEquals('string_: "a"', 'small <>')
    self.assertNotEquals('string_: "a"', 'small < strings: "b" >')
    self.assertNotEquals('string_: "a"', 'small < strings: "b" strings: "c" >')
    self.assertNotEquals('string_: "a" small <>', 'small <>')
    self.assertNotEquals('string_: "a" small <>', 'small < strings: "b" >')
    self.assertEquals('string_: "a" small <>', 'string_: "a" small <>')
    self.assertNotEquals('string_: "a" small < strings: "a" >',
                         'string_: "a" small <>')
    self.assertEquals('string_: "a" small < strings: "a" >',
                      'string_: "a" small < strings: "a" >')
    self.assertNotEquals('string_: "a" small < strings: "a" >',
                         'int64_: 1 small < strings: "a" >')
    self.assertNotEquals('string_: "a" small < strings: "a" >', 'int64_: 1')
    self.assertNotEquals('string_: "a"', 'int64_: 1 small < strings: "a" >')
    self.assertNotEquals('string_: "a" int64_: 0 small < strings: "a" >',
                         'int64_: 1 small < strings: "a" >')
    self.assertNotEquals('string_: "a" int64_: 1 small < strings: "a" >',
                         'string_: "a" int64_: 0 small < strings: "a" >')
    self.assertEquals('string_: "a" int64_: 0 small < strings: "a" >',
                      'string_: "a" int64_: 0 small < strings: "a" >')

  def testNestedMessage(self):
    self.assertNotEquals('medium <>', '')
    self.assertEquals('medium <>', 'medium <>')
    self.assertNotEquals('medium < smalls <> >', 'medium <>')
    self.assertEquals('medium < smalls <> >', 'medium < smalls <> >')
    self.assertNotEquals(
        'medium < smalls <> smalls <> >', 'medium < smalls <> >')
    self.assertEquals('medium < smalls <> smalls <> >',
                      'medium < smalls <> smalls <> >')

    self.assertNotEquals('medium < int32s: 0 >', 'medium < smalls <> >')

    self.assertNotEquals('medium < smalls < strings: "a"> >',
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
    self.assertNotEquals('string_: "a"                      ',
                         '             int64_: 1            ')
    self.assertNotEquals('string_: "a" int64_: 2            ',
                         '             int64_: 1            ')
    self.assertNotEquals('string_: "b" int64_: 1            ',
                         'string_: "a" int64_: 2            ')
    self.assertEquals('string_: "a" int64_: 1            ',
                      'string_: "a" int64_: 1            ')
    self.assertNotEquals('string_: "a" int64_: 1 float_: 0.0',
                         'string_: "a" int64_: 1            ')
    self.assertEquals('string_: "a" int64_: 1 float_: 0.0',
                      'string_: "a" int64_: 1 float_: 0.0')
    self.assertNotEquals('string_: "a" int64_: 1 float_: 0.1',
                         'string_: "a" int64_: 1 float_: 0.0')
    self.assertNotEquals('string_: "a" int64_: 2 float_: 0.0',
                         'string_: "a" int64_: 1 float_: 0.1')
    self.assertNotEquals('string_: "a"                      ',
                         '             int64_: 1 float_: 0.1')
    self.assertNotEquals('string_: "a"           float_: 0.0',
                         '             int64_: 1            ')
    self.assertNotEquals('string_: "b"           float_: 0.0',
                         'string_: "a" int64_: 1            ')

    self.assertNotEquals('string_: "a"',
                         'small < strings: "a" >')
    self.assertNotEquals('string_: "a" small < strings: "a" >',
                         'small < strings: "b" >')
    self.assertNotEquals('string_: "a" small < strings: "b" >',
                         'string_: "a" small < strings: "a" >')
    self.assertEquals('string_: "a" small < strings: "a" >',
                      'string_: "a" small < strings: "a" >')

    self.assertNotEquals('string_: "a" medium <>',
                         'string_: "a" small < strings: "a" >')
    self.assertNotEquals('string_: "a" medium < smalls <> >',
                         'string_: "a" small < strings: "a" >')
    self.assertNotEquals('medium <>', 'small < strings: "a" >')
    self.assertNotEquals('medium <> small <>', 'small < strings: "a" >')
    self.assertNotEquals('medium < smalls <> >', 'small < strings: "a" >')
    self.assertNotEquals('medium < smalls < strings: "a" > >',
                         'small < strings: "b" >')


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
  """Tests assertProtoEqual()."""
  def assertProtoEqual(self, a, b, **kwargs):
    if isinstance(a, six.string_types) and isinstance(b, six.string_types):
      a, b = LargePbs(a, b)
    compare.assertProtoEqual(self, a, b, **kwargs)

  def assertAll(self, a, **kwargs):
    """Checks that all possible asserts pass."""
    self.assertProtoEqual(a, a, **kwargs)

  def assertSameNotEqual(self, a, b):
    """Checks that assertProtoEqual() fails."""
    self.assertRaises(AssertionError, self.assertProtoEqual, a, b)

  def assertNone(self, a, b, message, **kwargs):
    """Checks that all possible asserts fail with the given message."""
    message = re.escape(textwrap.dedent(message))
    self.assertRaisesRegexp(AssertionError, message,
                            self.assertProtoEqual, a, b, **kwargs)

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
    compare.assertProtoEqual(
        self,
        """
          string_: 'abc'
          float_: 1.234
        """,
        pb)

  def testNormalizesNumbers(self):
    pb1 = compare_test_pb2.Large()
    pb1.int64_ = 4
    pb2 = compare_test_pb2.Large()
    pb2.int64_ = 4
    compare.assertProtoEqual(self, pb1, pb2)

  def testNormalizesFloat(self):
    pb1 = compare_test_pb2.Large()
    pb1.double_ = 4.0
    pb2 = compare_test_pb2.Large()
    pb2.double_ = 4
    compare.assertProtoEqual(self, pb1, pb2, normalize_numbers=True)

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

    self.assertSameNotEqual('int64s: 0', 'int64s: 0 int64s: 0')
    self.assertSameNotEqual('int64s: 0 int64s: 1',
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

    self.assertSameNotEqual(
        'medium: { smalls: { strings: "x" strings: "y" strings: "x" } }',
        'medium: { smalls: { strings: "y" strings: "x" } }')
    self.assertSameNotEqual(
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
                            self.assertProtoEqual,
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

    self.assertSameNotEqual('medium: { smalls: {} }',
                            'medium: { smalls: {} smalls: {} }')
    self.assertSameNotEqual('medium: { smalls: {} smalls: {} } medium: {}',
                            'medium: {} medium: {} medium: { smalls: {} }')
    self.assertSameNotEqual(
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


class MixinTests(compare.ProtoAssertions, googletest.TestCase):

  def testAssertEqualWithStringArg(self):
    pb = compare_test_pb2.Large()
    pb.string_ = 'abc'
    pb.float_ = 1.234
    self.assertProtoEqual(
        """
          string_: 'abc'
          float_: 1.234
        """,
        pb)


if __name__ == '__main__':
  googletest.main()
