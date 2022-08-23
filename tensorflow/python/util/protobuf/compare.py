# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Utility functions for comparing proto2 messages in Python.

ProtoEq() compares two proto2 messages for equality.

ClearDefaultValuedFields() recursively clears the fields that are set to their
default values. This is useful for comparing protocol buffers where the
semantics of unset fields and default valued fields are the same.

assertProtoEqual() is useful for unit tests.  It produces much more helpful
output than assertEqual() for proto2 messages, e.g. this:

  outer {
    inner {
-     strings: "x"
?               ^
+     strings: "y"
?               ^
    }
  }

...compared to the default output from assertEqual() that looks like this:

AssertionError: <my.Msg object at 0x9fb353c> != <my.Msg object at 0x9fb35cc>

Call it inside your unit test's googletest.TestCase subclasses like this:

  from tensorflow.python.util.protobuf import compare

  class MyTest(googletest.TestCase):
    ...
    def testXXX(self):
      ...
      compare.assertProtoEqual(self, a, b)

Alternatively:

  from tensorflow.python.util.protobuf import compare

  class MyTest(compare.ProtoAssertions, googletest.TestCase):
    ...
    def testXXX(self):
      ...
      self.assertProtoEqual(a, b)
"""

import difflib

from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format

from ..compat import collections_abc


def assertProtoEqual(self, a, b, check_initialized=True,  # pylint: disable=invalid-name
                     normalize_numbers=False, msg=None):
  """Fails with a useful error if a and b aren't equal.

  Comparison of repeated fields matches the semantics of
  unittest.TestCase.assertEqual(), ie order and extra duplicates fields matter.

  Args:
    self: googletest.TestCase
    a: proto2 PB instance, or text string representing one.
    b: proto2 PB instance -- message.Message or subclass thereof.
    check_initialized: boolean, whether to fail if either a or b isn't
      initialized.
    normalize_numbers: boolean, whether to normalize types and precision of
      numbers before comparison.
    msg: if specified, is used as the error message on failure.
  """
  pool = descriptor_pool.Default()
  if isinstance(a, str):
    a = text_format.Merge(a, b.__class__(), descriptor_pool=pool)

  for pb in a, b:
    if check_initialized:
      errors = pb.FindInitializationErrors()
      if errors:
        self.fail('Initialization errors: %s\n%s' % (errors, pb))
    if normalize_numbers:
      NormalizeNumberFields(pb)

  a_str = text_format.MessageToString(a, descriptor_pool=pool)
  b_str = text_format.MessageToString(b, descriptor_pool=pool)

  # Some Python versions would perform regular diff instead of multi-line
  # diff if string is longer than 2**16. We substitute this behavior
  # with a call to unified_diff instead to have easier-to-read diffs.
  # For context, see: https://bugs.python.org/issue11763.
  if len(a_str) < 2**16 and len(b_str) < 2**16:
    self.assertMultiLineEqual(a_str, b_str, msg=msg)
  else:
    diff = ''.join(
        difflib.unified_diff(a_str.splitlines(True), b_str.splitlines(True)))
    if diff:
      self.fail('%s :\n%s' % (msg, diff))


def NormalizeNumberFields(pb):
  """Normalizes types and precisions of number fields in a protocol buffer.

  Due to subtleties in the python protocol buffer implementation, it is possible
  for values to have different types and precision depending on whether they
  were set and retrieved directly or deserialized from a protobuf. This function
  normalizes integer values to ints and longs based on width, 32-bit floats to
  five digits of precision to account for python always storing them as 64-bit,
  and ensures doubles are floating point for when they're set to integers.

  Modifies pb in place. Recurses into nested objects.

  Args:
    pb: proto2 message.

  Returns:
    the given pb, modified in place.
  """
  for desc, values in pb.ListFields():
    is_repeated = True
    if desc.label != descriptor.FieldDescriptor.LABEL_REPEATED:
      is_repeated = False
      values = [values]

    normalized_values = None

    # We force 32-bit values to int and 64-bit values to long to make
    # alternate implementations where the distinction is more significant
    # (e.g. the C++ implementation) simpler.
    if desc.type in (descriptor.FieldDescriptor.TYPE_INT64,
                     descriptor.FieldDescriptor.TYPE_UINT64,
                     descriptor.FieldDescriptor.TYPE_SINT64):
      normalized_values = [int(x) for x in values]
    elif desc.type in (descriptor.FieldDescriptor.TYPE_INT32,
                       descriptor.FieldDescriptor.TYPE_UINT32,
                       descriptor.FieldDescriptor.TYPE_SINT32,
                       descriptor.FieldDescriptor.TYPE_ENUM):
      normalized_values = [int(x) for x in values]
    elif desc.type == descriptor.FieldDescriptor.TYPE_FLOAT:
      normalized_values = [round(x, 6) for x in values]
    elif desc.type == descriptor.FieldDescriptor.TYPE_DOUBLE:
      normalized_values = [round(float(x), 7) for x in values]

    if normalized_values is not None:
      if is_repeated:
        pb.ClearField(desc.name)
        getattr(pb, desc.name).extend(normalized_values)
      else:
        setattr(pb, desc.name, normalized_values[0])

    if (desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE or
        desc.type == descriptor.FieldDescriptor.TYPE_GROUP):
      if (desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE and
          desc.message_type.has_options and
          desc.message_type.GetOptions().map_entry):
        # This is a map, only recurse if the values have a message type.
        if (desc.message_type.fields_by_number[2].type ==
            descriptor.FieldDescriptor.TYPE_MESSAGE):
          for v in values.values():
            NormalizeNumberFields(v)
      else:
        for v in values:
          # recursive step
          NormalizeNumberFields(v)

  return pb


def _IsMap(value):
  return isinstance(value, collections_abc.Mapping)


def _IsRepeatedContainer(value):
  if isinstance(value, str):
    return False
  try:
    iter(value)
    return True
  except TypeError:
    return False


def ProtoEq(a, b):
  """Compares two proto2 objects for equality.

  Recurses into nested messages. Uses list (not set) semantics for comparing
  repeated fields, ie duplicates and order matter.

  Args:
    a: A proto2 message or a primitive.
    b: A proto2 message or a primitive.

  Returns:
    `True` if the messages are equal.
  """
  def Format(pb):
    """Returns a dictionary or unchanged pb bases on its type.

    Specifically, this function returns a dictionary that maps tag
    number (for messages) or element index (for repeated fields) to
    value, or just pb unchanged if it's neither.

    Args:
      pb: A proto2 message or a primitive.
    Returns:
      A dict or unchanged pb.
    """
    if isinstance(pb, message.Message):
      return dict((desc.number, value) for desc, value in pb.ListFields())
    elif _IsMap(pb):
      return dict(pb.items())
    elif _IsRepeatedContainer(pb):
      return dict(enumerate(list(pb)))
    else:
      return pb

  a, b = Format(a), Format(b)

  # Base case
  if not isinstance(a, dict) or not isinstance(b, dict):
    return a == b

  # This list performs double duty: it compares two messages by tag value *or*
  # two repeated fields by element, in order. the magic is in the format()
  # function, which converts them both to the same easily comparable format.
  for tag in sorted(set(a.keys()) | set(b.keys())):
    if tag not in a or tag not in b:
      return False
    else:
      # Recursive step
      if not ProtoEq(a[tag], b[tag]):
        return False

  # Didn't find any values that differed, so they're equal!
  return True


class ProtoAssertions:
  """Mix this into a googletest.TestCase class to get proto2 assertions.

  Usage:

  class SomeTestCase(compare.ProtoAssertions, googletest.TestCase):
    ...
    def testSomething(self):
      ...
      self.assertProtoEqual(a, b)

  See module-level definitions for method documentation.
  """

  # pylint: disable=invalid-name
  def assertProtoEqual(self, *args, **kwargs):
    return assertProtoEqual(self, *args, **kwargs)
