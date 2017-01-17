# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""## Functions for working with arbitrarily nested sequences of elements.

This module is used to perform any operations on nested structures, which can be
specified as sequences that contain non-sequence elements or other sequences.
The utilities here assume (and do not check) that the nested structures form a
'tree', i.e. no references in the structure of the input of these functions
should be recursive.

@@assert_same_structure
@@is_sequence
@@flatten
@@flatten_dict_items
@@pack_sequence_as
@@map_structure
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six


def _sequence_like(instance, args):
  """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, or a `namedtuple` class.
    args: elements to be converted to a sequence.

  Returns:
    `args` with the type of `instance`.
  """
  if (isinstance(instance, tuple) and
      hasattr(instance, "_fields") and
      isinstance(instance._fields, collections.Sequence) and
      all(isinstance(f, six.string_types) for f in instance._fields)):
    # This is a namedtuple
    return type(instance)(*args)
  else:
    # Not a namedtuple
    return type(instance)(args)


def _yield_flat_nest(nest):
  for n in nest:
    if is_sequence(n):
      for ni in _yield_flat_nest(n):
        yield ni
    else:
      yield n


def is_sequence(seq):
  """Returns a true if its input is a collections.Sequence (except strings).

  Args:
    seq: an input sequence.

  Returns:
    True if the sequence is a not a string and is a collections.Sequence.
  """
  return (isinstance(seq, collections.Sequence)
          and not isinstance(seq, six.string_types))


def flatten(nest):
  """Returns a flat sequence from a given nested structure.

  If `nest` is not a sequence, this returns a single-element list: `[nest]`.

  Args:
    nest: an arbitrarily nested structure or a scalar object.
      Note, numpy arrays are considered scalars.

  Returns:
    A Python list, the flattened version of the input.
  """
  return list(_yield_flat_nest(nest)) if is_sequence(nest) else [nest]


def _recursive_assert_same_structure(nest1, nest2):
  is_sequence_nest1 = is_sequence(nest1)
  if is_sequence_nest1 != is_sequence(nest2):
    raise ValueError(
        "The two structures don't have the same nested structure. "
        "First structure: %s, second structure: %s." % (nest1, nest2))

  if is_sequence_nest1:
    type_nest1 = type(nest1)
    type_nest2 = type(nest2)
    if type_nest1 != type_nest2:
      raise TypeError(
          "The two structures don't have the same sequence type. First "
          "structure has type %s, while second structure has type %s."
          % (type_nest1, type_nest2))

    for n1, n2 in zip(nest1, nest2):
      _recursive_assert_same_structure(n1, n2)


def assert_same_structure(nest1, nest2):
  """Asserts that two structures are nested in the same way.

  Args:
    nest1: an arbitrarily nested structure.
    nest2: an arbitrarily nested structure.

  Raises:
    ValueError: If the two structures do not have the same number of elements or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures.
  """
  len_nest1 = len(flatten(nest1)) if is_sequence(nest1) else 1
  len_nest2 = len(flatten(nest2)) if is_sequence(nest2) else 1
  if len_nest1 != len_nest2:
    raise ValueError("The two structures don't have the same number of "
                     "elements. First structure: %s, second structure: %s."
                     % (nest1, nest2))
  _recursive_assert_same_structure(nest1, nest2)


def flatten_dict_items(dictionary):
  """Returns a dictionary with flattened keys and values.

  This function flattens the keys and values of a dictionary, which can be
  arbitrarily nested structures, and returns the flattened version of such
  structures:

  ```python
  example_dictionary = {(4, 5, (6, 8)): ("a", "b", ("c", "d"))}
  result = {4: "a", 5: "b", 6: "c", 8: "d"}
  flatten_dict_items(example_dictionary) == result
  ```

  The input dictionary must satisfy two properties:

  1. Its keys and values should have the same exact nested structure.
  2. The set of all flattened keys of the dictionary must not contain repeated
     keys.

  Args:
    dictionary: the dictionary to zip

  Returns:
    The zipped dictionary.

  Raises:
    TypeError: If the input is not a dictionary.
    ValueError: If any key and value have not the same structure, or if keys are
      not unique.
  """
  if not isinstance(dictionary, dict):
    raise TypeError("input must be a dictionary")
  flat_dictionary = {}
  for i, v in six.iteritems(dictionary):
    if not is_sequence(i):
      if i in flat_dictionary:
        raise ValueError(
            "Could not flatten dictionary: key %s is not unique." % i)
      flat_dictionary[i] = v
    else:
      flat_i = flatten(i)
      flat_v = flatten(v)
      if len(flat_i) != len(flat_v):
        raise ValueError(
            "Could not flatten dictionary. Key had %d elements, but value had "
            "%d elements. Key: %s, value: %s."
            % (len(flat_i), len(flat_v), flat_i, flat_v))
      for new_i, new_v in zip(flat_i, flat_v):
        if new_i in flat_dictionary:
          raise ValueError(
              "Could not flatten dictionary: key %s is not unique."
              % (new_i))
        flat_dictionary[new_i] = new_v
  return flat_dictionary


def _packed_nest_with_indices(structure, flat, index):
  """Helper function for pack_nest_as.

  Args:
    structure: Substructure (tuple of elements and/or tuples) to mimic
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.

  Returns:
    The tuple (new_index, child), where:
      * new_index - the updated index into `flat` having processed `structure`.
      * packed - the subset of `flat` corresponding to `structure`,
                 having started at `index`, and packed into the same nested
                 format.

  Raises:
    ValueError: if `structure` contains more elements than `flat`
      (assuming indexing starts from `index`).
  """
  packed = []
  for s in structure:
    if is_sequence(s):
      new_index, child = _packed_nest_with_indices(s, flat, index)
      packed.append(_sequence_like(s, child))
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed


def pack_sequence_as(structure, flat_sequence):
  """Returns a given flattened sequence packed into a nest.

  If `structure` is a scalar, `flat_sequence` must be a single-element list;
  in this case the return value is `flat_sequence[0]`.

  Args:
    structure: tuple or list constructed of scalars and/or other tuples/lists,
      or a scalar.  Note: numpy arrays are considered scalars.
    flat_sequence: flat sequence to pack.

  Returns:
    packed: `flat_sequence` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If nest and structure have different element counts.
  """
  if not is_sequence(flat_sequence):
    raise TypeError("flat_sequence must be a sequence")

  if not is_sequence(structure):
    if len(flat_sequence) != 1:
      raise ValueError("Structure is a scalar but len(flat_sequence) == %d > 1"
                       % len(flat_sequence))
    return flat_sequence[0]

  flat_structure = flatten(structure)
  if len(flat_structure) != len(flat_sequence):
    raise ValueError(
        "Could not pack sequence. Structure had %d elements, but flat_sequence "
        "had %d elements.  Structure: %s, flat_sequence: %s."
        % (len(flat_structure), len(flat_sequence), structure, flat_sequence))

  _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
  return _sequence_like(structure, packed)


def map_structure(func, *structure):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(x[0], x[1], ...)` where x[i] is an entry in
  `structure[i]`.  All structures in `structure` must have the same arity,
  and the return value will contain the results in the same structure.

  Args:
    func: A callable that acceps as many arguments are there are structures.
    *structure: scalar, or tuple or list of constructed scalars and/or other
      tuples/lists, or scalars.  Note: numpy arrays are considered scalars.

  Returns:
    A new structure with the same arity as `structure`, whose values correspond
    to `func(x[0], x[1], ...)` where `x[i]` is a value in the corresponding
    location in `structure[i]`.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    ValueError: If no structure is provided or if the structures do not match
      each other by type.
  """
  if not callable(func):
    raise TypeError("func must be callable, got: %s" % func)

  if not structure:
    raise ValueError("Must provide at least one structure")

  for other in structure[1:]:
    assert_same_structure(structure[0], other)

  flat_structure = [flatten(s) for s in structure]
  entries = zip(*flat_structure)

  return pack_sequence_as(
      structure[0], [func(*x) for x in entries])
