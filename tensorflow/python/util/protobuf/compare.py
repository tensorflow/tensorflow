#!/usr/bin/python2.4

"""Utility functions for comparing proto2 messages in Python.

Proto2Cmp() is a cmp-style comparison function. It can be passed to sort(), etc.
See its docstring for details.

ClearDefaultValuedFields() recursively clears the fields that are set to their
default values. This is useful for comparing protocol buffers where the
semantics of unset fields and default valued fields are the same.

NormalizeRepeatedFields() sorts and optionally de-dupes repeated fields. This
is useful for treating repeated fields as sets instead of lists.

assertProto2Equal() and assertProto2SameElements() are useful for unit tests.
They produce much more helpful output than assertEqual() and friends for proto2
messages, e.g. this:

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

Call them inside your unit test's googletest.TestCase subclasses like this:

  from tensorflow.python.util.protobuf import compare

  class MyTest(googletest.TestCase):
    ...
    def testXXX(self):
      ...
      compare.assertProto2Equal(self, a, b)
      compare.assertProto2SameElements(self, a, c)

Alternatively:

  from tensorflow.python.util.protobuf import compare

  class MyTest(compare.Proto2Assertions, googletest.TestCase):
    ...
    def testXXX(self):
      ...
      self.assertProto2Equal(a, b)
      self.assertProto2SameElements(a, c)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import text_format
import six
from six.moves import xrange  # pylint: disable=redefined-builtin


def assertProto2Equal(self, a, b, check_initialized=True,
                      normalize_numbers=False, msg=None):
  """Fails with a useful error if a and b aren't equal.

  Comparison of repeated fields matches the semantics of
  unittest.TestCase.assertEqual(), ie order and extra duplicates fields matter.

  Args:
    self: googletest.TestCase
    a: proto2 PB instance, or text string representing one
    b: proto2 PB instance -- message.Message or subclass thereof
    check_initialized: boolean, whether to fail if either a or b isn't
      initialized
    normalize_numbers: boolean, whether to normalize types and precision of
      numbers before comparison.
    msg: if specified, is used as the error message on failure
  """
  if isinstance(a, six.string_types):
    a = text_format.Merge(a, b.__class__())

  for pb in a, b:
    if check_initialized:
      errors = pb.FindInitializationErrors()
      if errors:
        self.fail('Initialization errors: %s\n%s' % (errors, pb))
    if normalize_numbers:
      NormalizeNumberFields(pb)

  self.assertMultiLineEqual(text_format.MessageToString(a),
                            text_format.MessageToString(b),
                            msg=msg)


def assertProto2SameElements(self, a, b, number_matters=False,
                             check_initialized=True, normalize_numbers=False,
                             msg=None):
  """Fails with a useful error if a and b aren't equivalent.

  When comparing repeated fields, order doesn't matter and the number of times
  each element appears (ie duplicates) only matters if number_matters is True.

  By default, comparison of repeated fields follows set semantics and matches
  googletest.TestCase.assertSameElements(): neither order nor number of a given
  element matters.

  Args:
    self: googletest.TestCase
    a: proto2 PB instance, or text string representing one
    b: proto2 PB instance -- message.Message or subclass thereof
    number_matters: boolean, whether number of each elements must match
    check_initialized: boolean, whether to fail if either a or b isn't
      initialized
    normalize_numbers: boolean, whether to normalize types and precision of
      numbers before comparison.
    msg: if specified, is used as the error message on failure
  """
  if isinstance(a, six.string_types):
    a = text_format.Merge(a, b.__class__())
  else:
    a = copy.deepcopy(a)
  b = copy.deepcopy(b)
  for pb in a, b:
    NormalizeRepeatedFields(pb, dedupe=not number_matters)
  assertProto2Equal(
      self, a, b, check_initialized=check_initialized,
      normalize_numbers=normalize_numbers, msg=msg)


def assertProto2Contains(self, a, b,  # pylint: disable=invalid-name
                         number_matters=False, check_initialized=True,
                         msg=None):
  """Fails with a useful error if fields in a are not in b.

  Useful to test if expected fields are in b, allows tests to define
  expected fields in string format.

  Example:
    compare.assertProto2Contains('group { field: "value" }', test_pb2)

  Args:
    self: googletest.TestCase
    a: proto2 PB instance, or text string representing one
    b: proto2 PB instance
    number_matters: boolean, whether number of each field must match
    check_initialized: boolean, whether to fail if b isn't initialized
    msg: if specified, is used as the error message on failure
  """
  if isinstance(a, six.string_types):
    a = text_format.Merge(a, b.__class__())
  else:
    a = copy.deepcopy(a)
  completed_a = copy.deepcopy(b)
  completed_a.MergeFrom(a)
  assertProto2SameElements(self, completed_a, b, number_matters=number_matters,
                           check_initialized=check_initialized, msg=msg)


def ClearDefaultValuedFields(pb):
  """Clears all fields in a proto2 message that are set to their default values.

  The result has more compact text / json / binary representation. It's also
  easier to compare to other protos if the choice whether fields are not set or
  set to their default values doesn't change the proto buffer's semantics.

  Args:
    pb: A proto2 message.
  """
  for field, value in pb.ListFields():
    if field.type == field.TYPE_MESSAGE:
      if field.label == field.LABEL_REPEATED:
        for item in value:
          ClearDefaultValuedFields(item)
      else:
        ClearDefaultValuedFields(value)
        if field.label == field.LABEL_OPTIONAL and not value.ListFields():
          pb.ClearField(field.name)
    elif field.label == field.LABEL_OPTIONAL and value == field.default_value:
      pb.ClearField(field.name)


def NormalizeRepeatedFields(pb, dedupe=True):
  """Sorts all repeated fields and optionally removes duplicates.

  Modifies pb in place. Recurses into nested objects. Uses Proto2Cmp for
  sorting.

  Args:
    pb: proto2 message
    dedupe: boolean, whether to remove duplicates

  Returns: the given pb, modified in place
  """
  for desc, values in pb.ListFields():
    if desc.label is not descriptor.FieldDescriptor.LABEL_REPEATED:
      values = [values]

    if (desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE and
        desc.message_type.has_options and
        desc.message_type.GetOptions().map_entry):
      # This is a map, only recurse if the values have a message type.
      if (desc.message_type.fields_by_number[2].type ==
          descriptor.FieldDescriptor.TYPE_MESSAGE):
        for v in six.itervalues(values):
          NormalizeRepeatedFields(v, dedupe=dedupe)
    else:
      if (desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE or
          desc.type == descriptor.FieldDescriptor.TYPE_GROUP):
        for v in values:
          # recursive step
          NormalizeRepeatedFields(v, dedupe=dedupe)

      values.sort(Proto2Cmp)

      if dedupe:
        # De-dupe in place. Can't use set, etc. because messages aren't
        # hashable.  This is a heavily discussed toy problem. the code below is
        # a simplified version of http://code.activestate.com/recipes/52560/
        # and it requires that values is sorted.
        for i in xrange(len(values) - 1, 0, -1):
          if values[i] == values[i - 1]:
            del values[i]

  return pb


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
    pb: proto2 message

  Returns:
    the given pb, modified in place
  """
  for desc, values in pb.ListFields():
    is_repeated = True
    if desc.label is not descriptor.FieldDescriptor.LABEL_REPEATED:
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
          for v in six.itervalues(values):
            NormalizeNumberFields(v)
      else:
        for v in values:
          # recursive step
          NormalizeNumberFields(v)

  return pb


def _IsRepeatedContainer(value):
  if isinstance(value, six.string_types):
    return False
  try:
    iter(value)
    return True
  except TypeError:
    return False


def Proto2Cmp(a, b):
  """Compares two proto2 objects field by field, in ascending tag order.

  Recurses into nested messages. Uses list (not set) semantics for comparing
  repeated fields, ie duplicates and order matter. If one field is a prefix of
  the other, the longer field is greater.

  This function is intended to be used as a python cmp function, e.g. in sort.

  Ordering fields by tag number has precedent in other google code, but it's
  still somewhat arbitrary. The main value is to provide *some* stable ordering
  for proto2 messages.

  This would be easier as a__cmp__ method or set of __le__, __gt__, etc methods
  in the proto2 Message class itself. That would take a little more care,
  though, and probably some significant debate over whether they should exist at
  all, so this was easier.

  Args:
    a, b: proto2 messages or primitives

  Returns: integer > 0 if a > b, < 0 if a < b, 0 if a == b
  """
  def Format(pb):
    """Returns a dictionary that maps tag number (for messages) or element index
    (for repeated fields) to value, or just pb unchanged if it's neither."""
    if isinstance(pb, message.Message):
      return dict((desc.number, value) for desc, value in pb.ListFields())
    elif _IsRepeatedContainer(pb):
      return dict(enumerate(pb))
    else:
      return pb

  a, b = Format(a), Format(b)

  # base case
  if not isinstance(a, dict) or not isinstance(b, dict):
    return cmp(a, b)

  # this list performs double duty: it compares two messages by tag value *or*
  # two repeated fields by element, in order. the magic is in the format()
  # function, which converts them both to the same easily comparable format.
  for tag in sorted(set(a.keys()) | set(b.keys())):
    if tag not in a:
      return -1  # b is greater
    elif tag not in b:
      return 1   # a is greater
    else:
      # recursive step
      cmped = Proto2Cmp(a[tag], b[tag])
      if cmped != 0:
        return cmped

  # didn't find any values that differed, so they're equal!
  return 0


class Proto2Assertions(object):
  """Mix this into a googletest.TestCase class to get proto2 assertions.

  Usage:

  class SomeTestCase(compare.Proto2Assertions, googletest.TestCase):
    ...
    def testSomething(self):
      ...
      self.assertProto2Equal(a, b)

  See module-level definitions for method documentation.
  """

  # pylint: disable=invalid-name
  def assertProto2Equal(self, *args, **kwargs):
    return assertProto2Equal(self, *args, **kwargs)

  def assertProto2SameElements(self, *args, **kwargs):
    return assertProto2SameElements(self, *args, **kwargs)

  def assertProto2Contains(self, *args, **kwargs):
    return assertProto2Contains(self, *args, **kwargs)
