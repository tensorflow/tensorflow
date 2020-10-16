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

This module can perform operations on nested structures. A nested structure is a
Python collection that can contain further collections as well as other objects
called atoms. Note that numpy arrays are considered atoms.

nest recognizes the following types of collections:
  1.tuple
  2.namedtuple
  3.dict
  4.orderedDict
  5.MutableMapping
  6.attr.s

attr.s decorated classes (http://www.attrs.org) are also supported, in the
same way as `namedtuple`.

The utilities here assume (and do not check) that the nested structures form a
'tree', i.e., no references in the structure of the input of these functions
should be recursive.

Example structures: `((3, 4), 5, (6, 7, (9, 10), 8))`, `(np.array(0),
  (np.array([3, 4]), tf.constant([3, 4])))`
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as _collections

import six as _six
import wrapt as _wrapt

from tensorflow.python import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.platform import tf_logging


_SHALLOW_TREE_HAS_INVALID_KEYS = (
    "The shallow_tree's keys are not a subset of the input_tree's keys. The "
    "shallow_tree has the following keys that are not in the input_tree: {}.")

_STRUCTURES_HAVE_MISMATCHING_TYPES = (
    "The two structures don't have the same sequence type. Input structure has "
    "type {input_type}, while shallow structure has type {shallow_type}.")

_STRUCTURES_HAVE_MISMATCHING_LENGTHS = (
    "The two structures don't have the same sequence length. Input "
    "structure has length {input_length}, while shallow structure has length "
    "{shallow_length}."
)

_INPUT_TREE_SMALLER_THAN_SHALLOW_TREE = (
    "The input_tree has fewer elements than the shallow_tree. Input structure "
    "has length {input_size}, while shallow structure has length "
    "{shallow_size}.")

_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ = (
    "If shallow structure is a sequence, input must also be a sequence. "
    "Input has type: {}.")


def _get_attrs_items(obj):
  """Returns a list of (name, value) pairs from an attrs instance.

  The list will be sorted by name.

  Args:
    obj: an object.

  Returns:
    A list of (attr_name, attr_value) pairs, sorted by attr_name.
  """
  attrs = getattr(obj.__class__, "__attrs_attrs__")
  attr_names = (a.name for a in attrs)
  return [(attr_name, getattr(obj, attr_name)) for attr_name in attr_names]


def _sorted(dict_):
  """Returns a sorted list of the dict keys, with error if keys not sortable."""
  try:
    return sorted(dict_.keys())
  except TypeError:
    raise TypeError("nest only supports dicts with sortable keys.")


def _is_namedtuple(instance, strict=False):
  """Returns True iff `instance` is a `namedtuple`.

  Args:
    instance: An instance of a Python object.
    strict: If True, `instance` is considered to be a `namedtuple` only if
        it is a "plain" namedtuple. For instance, a class inheriting
        from a `namedtuple` will be considered to be a `namedtuple`
        iff `strict=False`.

  Returns:
    True if `instance` is a `namedtuple`.
  """
  return _pywrap_utils.IsNamedtuple(instance, strict)


# See the swig file (util.i) for documentation.
_is_mapping_view = _pywrap_utils.IsMappingView
_is_attrs = _pywrap_utils.IsAttrs
_is_composite_tensor = _pywrap_utils.IsCompositeTensor
_is_type_spec = _pywrap_utils.IsTypeSpec
_is_mutable_mapping = _pywrap_utils.IsMutableMapping
_is_mapping = _pywrap_utils.IsMapping


def _sequence_like(instance, args):
  """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`,
        `collections.OrderedDict`, or `composite_tensor.Composite_Tensor`
        or `type_spec.TypeSpec`.
    args: elements to be converted to the `instance` type.

  Returns:
    `args` with the type of `instance`.
  """
  if _is_mutable_mapping(instance):
    # Pack dictionaries in a deterministic order by sorting the keys.
    # Notice this means that we ignore the original order of `OrderedDict`
    # instances. This is intentional, to avoid potential bugs caused by mixing
    # ordered and plain dicts (e.g., flattening a dict but using a
    # corresponding `OrderedDict` to pack it back).
    result = dict(zip(_sorted(instance), args))
    instance_type = type(instance)
    if instance_type == _collections.defaultdict:
      d = _collections.defaultdict(instance.default_factory)
    else:
      d = instance_type()
    for key in instance:
      d[key] = result[key]
    return d
  elif _is_mapping(instance):
    result = dict(zip(_sorted(instance), args))
    instance_type = type(instance)
    tf_logging.log_first_n(
        tf_logging.WARN, "Mapping types may not work well with tf.nest. Prefer"
        " using MutableMapping for {}".format(instance_type), 1)
    try:
      return instance_type((key, result[key]) for key in instance)
    except TypeError as err:
      raise TypeError("Error creating an object of type {} like {}. Note that "
                      "it must accept a single positional argument "
                      "representing an iterable of key-value pairs, in "
                      "addition to self. Cause: {}".format(
                          type(instance), instance, err))
  elif _is_mapping_view(instance):
    # We can't directly construct mapping views, so we create a list instead
    return list(args)
  elif _is_namedtuple(instance) or _is_attrs(instance):
    if isinstance(instance, _wrapt.ObjectProxy):
      instance_type = type(instance.__wrapped__)
    else:
      instance_type = type(instance)
    return instance_type(*args)
  elif _is_composite_tensor(instance):
    assert len(args) == 1
    spec = instance._type_spec  # pylint: disable=protected-access
    return spec._from_components(args[0])  # pylint: disable=protected-access
  elif _is_type_spec(instance):
    # Pack a CompositeTensor's components according to a TypeSpec.
    assert len(args) == 1
    return instance._from_components(args[0])  # pylint: disable=protected-access
  elif isinstance(instance, _six.moves.range):
    return _sequence_like(list(instance), args)
  elif isinstance(instance, _wrapt.ObjectProxy):
    # For object proxies, first create the underlying type and then re-wrap it
    # in the proxy type.
    return type(instance)(_sequence_like(instance.__wrapped__, args))
  else:
    # Not a namedtuple
    return type(instance)(args)


def _yield_value(iterable):
  for _, v in _yield_sorted_items(iterable):
    yield v


def _yield_sorted_items(iterable):
  """Yield (key, value) pairs for `iterable` in a deterministic order.

  For Sequences, the key will be an int, the array index of a value.
  For Mappings, the key will be the dictionary key.
  For objects (e.g. namedtuples), the key will be the attribute name.

  In all cases, the keys will be iterated in sorted order.

  Args:
    iterable: an iterable.

  Yields:
    The iterable's (key, value) pairs, in order of sorted keys.
  """
  # Ordered to check common structure types (list, tuple, dict) first.
  if isinstance(iterable, list):
    for item in enumerate(iterable):
      yield item
  # namedtuples handled separately to avoid expensive namedtuple check.
  elif type(iterable) == tuple:  # pylint: disable=unidiomatic-typecheck
    for item in enumerate(iterable):
      yield item
  elif isinstance(iterable, (dict, _collections_abc.Mapping)):
    # Iterate through dictionaries in a deterministic order by sorting the
    # keys. Notice this means that we ignore the original order of `OrderedDict`
    # instances. This is intentional, to avoid potential bugs caused by mixing
    # ordered and plain dicts (e.g., flattening a dict but using a
    # corresponding `OrderedDict` to pack it back).
    for key in _sorted(iterable):
      yield key, iterable[key]
  elif _is_attrs(iterable):
    for item in _get_attrs_items(iterable):
      yield item
  elif _is_namedtuple(iterable):
    for field in iterable._fields:
      yield field, getattr(iterable, field)
  elif _is_composite_tensor(iterable):
    type_spec = iterable._type_spec  # pylint: disable=protected-access
    yield type_spec.value_type.__name__, type_spec._to_components(iterable)  # pylint: disable=protected-access
  elif _is_type_spec(iterable):
    # Note: to allow CompositeTensors and their TypeSpecs to have matching
    # structures, we need to use the same key string here.
    yield iterable.value_type.__name__, iterable._component_specs  # pylint: disable=protected-access
  else:
    for item in enumerate(iterable):
      yield item


# See the swig file (util.i) for documentation.
is_sequence = _pywrap_utils.IsSequence


# See the swig file (util.i) for documentation.
is_sequence_or_composite = _pywrap_utils.IsSequenceOrComposite


@tf_export("nest.is_nested")
def is_nested(seq):
  """Returns true if its input is a collections.abc.Sequence (except strings).

  Args:
    seq: an input sequence.

  Returns:
    True if the sequence is a not a string and is a collections.abc.Sequence
    or a dict.
  """
  return is_sequence(seq)


@tf_export("nest.flatten")
def flatten(structure, expand_composites=False):
  """Returns a flat list from a given nested structure.

  If nest is not a structure , tuple (or a namedtuple), dict, or an attrs class,
  then returns a single-element list:
    [nest].

  In the case of dict instances, the sequence consists of the values, sorted by
  key to ensure deterministic behavior. This is true also for OrderedDict
  instances: their sequence order is ignored, the sorting order of keys is used
  instead. The same convention is followed in pack_sequence_as. This correctly
  repacks dicts and OrderedDicts after they have been flattened, and also allows
  flattening an OrderedDict and then repacking it back using a corresponding
  plain dict, or vice-versa. Dictionaries with non-sortable keys cannot be
  flattened.

  Users must not modify any collections used in nest while this function is
  running.

  Examples:

  1. Python dict (ordered by key):

  >>> dict = { "key3": "value3", "key1": "value1", "key2": "value2" }
  >>> tf.nest.flatten(dict)
  ['value1', 'value2', 'value3']

  2. For a nested python tuple:

  >>> tuple = ((1.0, 2.0), (3.0, 4.0, 5.0), (6.0))
  >>> tf.nest.flatten(tuple)
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

  3. Numpy array (will not flatten):

  >>> array = np.array([[1, 2], [3, 4]])
  >>> tf.nest.flatten(array)
      [array([[1, 2],
              [3, 4]])]


  4. `tf.Tensor` (will not flatten):

  >>> tensor = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
  >>> tf.nest.flatten(tensor)
      [<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
        array([[1., 2., 3.],
               [4., 5., 6.],
               [7., 8., 9.]], dtype=float32)>]

  Args:
    structure: an arbitrarily nested structure. Note, numpy arrays are
      considered atoms and are not flattened.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A Python list, the flattened version of the input.

  Raises:
    TypeError: The nest is or contains a dict with non-sortable keys.
  """
  if structure is None:
    return [None]
  expand_composites = bool(expand_composites)
  return _pywrap_utils.Flatten(structure, expand_composites)


# See the swig file (util.i) for documentation.
_same_namedtuples = _pywrap_utils.SameNamedtuples


class _DotString(object):

  __slots__ = []

  def __str__(self):
    return "."

  def __repr__(self):
    return "."


_DOT = _DotString()


@tf_export("nest.assert_same_structure")
def assert_same_structure(nest1, nest2, check_types=True,
                          expand_composites=False):
  """Asserts that two structures are nested in the same way.

  Note that namedtuples with identical name and fields are always considered
  to have the same shallow structure (even with `check_types=True`).
  For instance, this code will print `True`:

  ```python
  def nt(a, b):
    return collections.namedtuple('foo', 'a b')(a, b)
  print(assert_same_structure(nt(0, 1), nt(2, 3)))
  ```

  Args:
    nest1: an arbitrarily nested structure.
    nest2: an arbitrarily nested structure.
    check_types: if `True` (default) types of sequences are checked as well,
      including the keys of dictionaries. If set to `False`, for example a
      list and a tuple of objects will look the same if they have the same
      size. Note that namedtuples with identical name and fields are always
      considered to have the same shallow structure. Two types will also be
      considered the same if they are both list subtypes (which allows "list"
      and "_ListWrapper" from trackable dependency tracking to compare
      equal).
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Raises:
    ValueError: If the two structures do not have the same number of elements or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures. Only possible if `check_types` is `True`.
  """
  # Convert to bool explicitly as otherwise pybind will not be able# to handle
  # type mismatch message correctly. See GitHub issue 42329 for details.
  check_types = bool(check_types)
  expand_composites = bool(expand_composites)
  try:
    _pywrap_utils.AssertSameStructure(nest1, nest2, check_types,
                                      expand_composites)
  except (ValueError, TypeError) as e:
    str1 = str(map_structure(lambda _: _DOT, nest1))
    str2 = str(map_structure(lambda _: _DOT, nest2))
    raise type(e)("%s\n"
                  "Entire first structure:\n%s\n"
                  "Entire second structure:\n%s"
                  % (str(e), str1, str2))


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
    ValueError: If any key and value do not have the same structure layout, or
    if keys are not unique.
  """
  if not isinstance(dictionary, (dict, _collections_abc.Mapping)):
    raise TypeError("input must be a dictionary")
  flat_dictionary = {}
  for i, v in _six.iteritems(dictionary):
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


def _packed_nest_with_indices(structure, flat, index, is_seq, sequence_fn=None):
  """Helper function for pack_sequence_as.

  Args:
    structure: Substructure (list / tuple / dict) to mimic.
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.
    is_seq: Function used to test if a value should be treated as a sequence.
    sequence_fn: Function used to generate a new sequence instance.

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
  sequence_fn = sequence_fn or _sequence_like
  for s in _yield_value(structure):
    if is_seq(s):
      new_index, child = _packed_nest_with_indices(s, flat, index, is_seq,
                                                   sequence_fn)
      packed.append(sequence_fn(s, child))
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed


def _pack_sequence_as(structure, flat_sequence, expand_composites,
                      sequence_fn=None):
  """Implements sequence packing, with the option to alter the structure."""
  is_seq = is_sequence_or_composite if expand_composites else is_sequence
  sequence_fn = sequence_fn or _sequence_like
  def truncate(value, length):
    value_str = str(value)
    return value_str[:length] + (value_str[length:] and "...")

  if not is_seq(flat_sequence):
    raise TypeError(
        "Attempted to pack value:\n  {}\ninto a sequence, but found "
        "incompatible type `{}` instead."
        .format(truncate(flat_sequence, 100), type(flat_sequence)))

  if not is_seq(structure):
    if len(flat_sequence) != 1:
      raise ValueError(
          "The target structure is of type `{}`\n  {}\nHowever the input "
          "structure is a sequence ({}) of length {}.\n  {}\nnest cannot "
          "guarantee that it is safe to map one to the other.".format(
              type(structure), truncate(structure, 100), type(flat_sequence),
              len(flat_sequence), truncate(flat_sequence, 100)))
    return flat_sequence[0]

  try:
    final_index, packed = _packed_nest_with_indices(structure, flat_sequence,
                                                    0, is_seq, sequence_fn)
    if final_index < len(flat_sequence):
      raise IndexError
  except IndexError:
    flat_structure = flatten(structure)
    if len(flat_structure) != len(flat_sequence):
      raise ValueError(
          "Could not pack sequence. Structure had %d elements, but "
          "flat_sequence had %d elements.  Structure: %s, flat_sequence: %s." %
          (len(flat_structure), len(flat_sequence), structure, flat_sequence))
  return sequence_fn(structure, packed)


@tf_export("nest.pack_sequence_as")
def pack_sequence_as(structure, flat_sequence, expand_composites=False):
  """Returns a given flattened sequence packed into a given structure.

  If `structure` is a scalar, `flat_sequence` must be a single-element list;
  in this case the return value is `flat_sequence[0]`.

  If `structure` is or contains a dict instance, the keys will be sorted to
  pack the flat sequence in deterministic order. This is true also for
  `OrderedDict` instances: their sequence order is ignored, the sorting order of
  keys is used instead. The same convention is followed in `flatten`.
  This correctly repacks dicts and `OrderedDict`s after they have been
  flattened, and also allows flattening an `OrderedDict` and then repacking it
  back using a corresponding plain dict, or vice-versa.
  Dictionaries with non-sortable keys cannot be flattened.

  Args:
    structure: Nested structure, whose structure is given by nested lists,
      tuples, and dicts. Note: numpy arrays and strings are considered
      scalars.
    flat_sequence: flat sequence to pack.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    packed: `flat_sequence` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If `flat_sequence` and `structure` have different
      element counts.
    TypeError: `structure` is or contains a dict with non-sortable keys.
  """
  return _pack_sequence_as(structure, flat_sequence, expand_composites)


@tf_export("nest.map_structure")
def map_structure(func, *structure, **kwargs):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(x[0], x[1], ...)` where x[i] is an entry in
  `structure[i]`.  All structures in `structure` must have the same arity,
  and the return value will contain results with the same structure layout.

  Examples:

  1. A single Python dict:

  >>> a = {"hello": 24, "world": 76}
  >>> tf.nest.map_structure(lambda p: p * 2, a)
  {'hello': 48, 'world': 152}

  2. Multiple Python dictionaries:

  >>> d1 = {"hello": 24, "world": 76}
  >>> d2 = {"hello": 36, "world": 14}
  >>> tf.nest.map_structure(lambda p1, p2: p1 + p2, d1, d2)
  {'hello': 60, 'world': 90}

  Args:
    func: A callable that accepts as many arguments as there are structures.
    *structure: scalar, or tuple or dict or list of constructed scalars and/or
      other tuples/lists, or scalars.  Note: numpy arrays are considered as
      scalars.
    **kwargs: Valid keyword args are:

      * `check_types`: If set to `True` (default) the types of
        iterables within the structures have to be same (e.g.
        `map_structure(func, [1], (1,))` raises a `TypeError`
        exception). To allow this set this argument to `False`.
        Note that namedtuples with identical name and fields are always
        considered to have the same shallow structure.
      * `expand_composites`: If set to `True`, then composite tensors such
        as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into
        their component tensors.  If `False` (the default), then composite
        tensors are not expanded.

  Returns:
    A new structure with the same arity as `structure`, whose values correspond
    to `func(x[0], x[1], ...)` where `x[i]` is a value in the corresponding
    location in `structure[i]`. If there are different sequence types and
    `check_types` is `False` the sequence types of the first structure will be
    used.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    ValueError: If no structure is provided or if the structures do not match
      each other by type.
    ValueError: If wrong keyword arguments are provided.
  """
  if not callable(func):
    raise TypeError("func must be callable, got: %s" % func)

  if not structure:
    raise ValueError("Must provide at least one structure")

  check_types = kwargs.pop("check_types", True)
  expand_composites = kwargs.pop("expand_composites", False)

  if kwargs:
    raise ValueError(
        "Only valid keyword arguments are `check_types` and "
        "`expand_composites`, not: `%s`" % ("`, `".join(kwargs.keys())))

  for other in structure[1:]:
    assert_same_structure(structure[0], other, check_types=check_types,
                          expand_composites=expand_composites)

  flat_structure = (flatten(s, expand_composites) for s in structure)
  entries = zip(*flat_structure)

  return pack_sequence_as(
      structure[0], [func(*x) for x in entries],
      expand_composites=expand_composites)


def map_structure_with_paths(func, *structure, **kwargs):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(path, x[0], x[1], ..., **kwargs)` where x[i] is an entry in
  `structure[i]` and `path` is the common path to x[i] in the structures.  All
  structures in `structure` must have the same arity, and the return value will
  contain the results with the same structure layout. Special kwarg
  `check_types` determines whether the types of iterables within the structure
  must be the same-- see **kwargs definition below.

  Args:
    func: A callable with the signature func(path, *values, **kwargs) that is
      evaluated on the leaves of the structure.
    *structure: A variable number of compatible structures to process.
    **kwargs: Optional kwargs to be passed through to func. Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.,
      `map_structure(func, [1], (1,))` raises a `TypeError` exception). By
      default, the types must match. To allow iteration over structures of
      different types (but common arity), set this kwarg to `False`.

  Returns:
    A structure of the same form as the input structures whose leaves are the
    result of evaluating func on corresponding leaves of the input structures.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    TypeError: If `check_types` is not `False` and the two structures differ in
      the type of sequence in any of their substructures.
    ValueError: If no structures are provided.
  """
  def wrapper_func(tuple_path, *inputs, **kwargs):
    string_path = "/".join(str(s) for s in tuple_path)
    return func(string_path, *inputs, **kwargs)

  return map_structure_with_tuple_paths_up_to(structure[0],
                                              wrapper_func,
                                              *structure,
                                              **kwargs)


def map_structure_with_tuple_paths(func, *structure, **kwargs):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(tuple_path, x[0], x[1], ..., **kwargs)` where `x[i]` is an entry
  in `structure[i]` and `tuple_path` is a tuple of indices and/or dictionary
  keys (as returned by `nest.yield_flat_paths`), which uniquely specifies the
  common path to x[i] in the structures. All structures in `structure` must have
  the same arity, and the return value will contain the results in the same
  structure. Special kwarg `check_types` determines whether the types of
  iterables within the structure must be the same-- see **kwargs definition
  below.

  Args:
    func: A callable with the signature `func(tuple_path, *values, **kwargs)`
      that is evaluated on the leaves of the structure.
    *structure: A variable number of compatible structures to process.
    **kwargs: Optional kwargs to be passed through to func. Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.
      `map_structure(func, [1], (1,))` raises a `TypeError` exception). To allow
      this set this argument to `False`.

  Returns:
    A structure of the same form as the input structures whose leaves are the
    result of evaluating func on corresponding leaves of the input structures.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    TypeError: If `check_types` is not `False` and the two structures differ in
      the type of sequence in any of their substructures.
    ValueError: If no structures are provided.
  """
  return map_structure_with_tuple_paths_up_to(structure[0],
                                              func,
                                              *structure,
                                              **kwargs)


def _yield_flat_up_to(shallow_tree, input_tree, is_seq, path=()):
  """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

  Args:
    shallow_tree: Nested structure. Traverse no further than its leaf nodes.
    input_tree: Nested structure. Return the paths and values from this tree.
      Must have the same upper structure as shallow_tree.
    is_seq: Function used to test if a value should be treated as a sequence.
    path: Tuple. Optional argument, only used when recursing. The path from the
      root of the original shallow_tree, down to the root of the shallow_tree
      arg of this recursive call.

  Yields:
    Pairs of (path, value), where path the tuple path of a leaf node in
    shallow_tree, and value is the value of the corresponding node in
    input_tree.
  """
  if not is_seq(shallow_tree):
    yield (path, input_tree)
  else:
    input_tree = dict(_yield_sorted_items(input_tree))
    for shallow_key, shallow_subtree in _yield_sorted_items(shallow_tree):
      subpath = path + (shallow_key,)
      input_subtree = input_tree[shallow_key]
      for leaf_path, leaf_value in _yield_flat_up_to(shallow_subtree,
                                                     input_subtree, is_seq,
                                                     path=subpath):
        yield (leaf_path, leaf_value)


def assert_shallow_structure(shallow_tree,
                             input_tree,
                             check_types=True,
                             expand_composites=False):
  """Asserts that `shallow_tree` is a shallow structure of `input_tree`.

  That is, this function tests if the `input_tree` structure can be created from
  the `shallow_tree` structure by replacing its leaf nodes with deeper
  tree structures.

  Examples:

  The following code will raise an exception:
  ```python
    shallow_tree = {"a": "A", "b": "B"}
    input_tree = {"a": 1, "c": 2}
    assert_shallow_structure(shallow_tree, input_tree)
  ```

  The following code will raise an exception:
  ```python
    shallow_tree = ["a", "b"]
    input_tree = ["c", ["d", "e"], "f"]
    assert_shallow_structure(shallow_tree, input_tree)
  ```

  Args:
    shallow_tree: an arbitrarily nested structure.
    input_tree: an arbitrarily nested structure.
    check_types: if `True` (default) the sequence types of `shallow_tree` and
      `input_tree` have to be the same. Note that even with check_types==True,
      this function will consider two different namedtuple classes with the same
      name and _fields attribute to be the same class.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.
  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`. Only raised if `check_types` is `True`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  is_seq = is_sequence_or_composite if expand_composites else is_sequence
  if is_seq(shallow_tree):
    if not is_seq(input_tree):
      raise TypeError(
          "If shallow structure is a sequence, input must also be a sequence. "
          "Input has type: %s." % type(input_tree))

    if isinstance(shallow_tree, _wrapt.ObjectProxy):
      shallow_type = type(shallow_tree.__wrapped__)
    else:
      shallow_type = type(shallow_tree)

    if check_types and not isinstance(input_tree, shallow_type):
      # Duck-typing means that nest should be fine with two different
      # namedtuples with identical name and fields.
      shallow_is_namedtuple = _is_namedtuple(shallow_tree, False)
      input_is_namedtuple = _is_namedtuple(input_tree, False)
      if shallow_is_namedtuple and input_is_namedtuple:
        if not _same_namedtuples(shallow_tree, input_tree):
          raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
              input_type=type(input_tree),
              shallow_type=type(shallow_tree)))

      elif ((_is_composite_tensor(shallow_tree) or
             _is_composite_tensor(input_tree)) and
            (_is_type_spec(shallow_tree) or _is_type_spec(input_tree))):
        pass  # Compatibility will be checked below.

      elif not (isinstance(shallow_tree, _collections_abc.Mapping) and
                isinstance(input_tree, _collections_abc.Mapping)):
        raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            input_type=type(input_tree),
            shallow_type=type(shallow_tree)))

    if _is_composite_tensor(shallow_tree) or _is_composite_tensor(input_tree):
      if not (
          (_is_composite_tensor(input_tree) or _is_type_spec(input_tree)) and
          (_is_composite_tensor(shallow_tree) or _is_type_spec(shallow_tree))):
        raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            input_type=type(input_tree),
            shallow_type=type(shallow_tree)))
      type_spec_1 = (shallow_tree if _is_type_spec(shallow_tree) else
                     shallow_tree._type_spec)  # pylint: disable=protected-access
      type_spec_2 = (input_tree if _is_type_spec(input_tree) else
                     input_tree._type_spec)  # pylint: disable=protected-access
      try:
        _ = type_spec_1.most_specific_compatible_type(type_spec_2)
      except (TypeError, ValueError) as e:
        raise ValueError(
            "Incompatible CompositeTensor TypeSpecs: %s vs. %s -- %s" %
            (type_spec_1, type_spec_2, e))

    elif _is_type_spec(shallow_tree):
      if not _is_type_spec(input_tree):
        raise TypeError("If shallow structure is a TypeSpec, input must also "
                        "be a TypeSpec.  Input has type: %s."
                        % type(input_tree))
    else:
      if len(input_tree) != len(shallow_tree):
        raise ValueError(
            _STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
                input_length=len(input_tree), shallow_length=len(shallow_tree)))
      elif len(input_tree) < len(shallow_tree):
        raise ValueError(
            _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE.format(
                input_size=len(input_tree), shallow_size=len(shallow_tree)))

    if isinstance(shallow_tree, _collections_abc.Mapping):
      absent_keys = set(shallow_tree) - set(input_tree)
      if absent_keys:
        raise ValueError(_SHALLOW_TREE_HAS_INVALID_KEYS
                         .format(sorted(absent_keys)))

    for shallow_branch, input_branch in zip(_yield_value(shallow_tree),
                                            _yield_value(input_tree)):
      assert_shallow_structure(shallow_branch, input_branch,
                               check_types=check_types,
                               expand_composites=expand_composites)


def flatten_up_to(shallow_tree, input_tree, check_types=True,
                  expand_composites=False):
  """Flattens `input_tree` up to `shallow_tree`.

  Any further depth in structure in `input_tree` is retained as elements in the
  partially flatten output.

  If `shallow_tree` and `input_tree` are not sequences, this returns a
  single-element list: `[input_tree]`.

  Use Case:

  Sometimes we may wish to partially flatten a nested sequence, retaining some
  of the nested structure. We achieve this by specifying a shallow structure,
  `shallow_tree`, we wish to flatten up to.

  The input, `input_tree`, can be thought of as having the same structure layout
  as `shallow_tree`, but with leaf nodes that are themselves tree structures.

  Examples:

  ```python
  input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
  shallow_tree = [[True, True], [False, True]]

  flattened_input_tree = flatten_up_to(shallow_tree, input_tree)
  flattened_shallow_tree = flatten_up_to(shallow_tree, shallow_tree)

  # Output is:
  # [[2, 2], [3, 3], [4, 9], [5, 5]]
  # [True, True, False, True]
  ```

  ```python
  input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
  shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]

  input_tree_flattened_as_shallow_tree = flatten_up_to(shallow_tree, input_tree)
  input_tree_flattened = flatten(input_tree)

  # Output is:
  # [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
  # ['a', 1, 'b', 2, 'c', 3, 'd', 4]
  ```

  Non-Sequence Edge Cases:

  ```python
  flatten_up_to(0, 0)  # Output: [0]
  flatten_up_to(0, [0, 1, 2])  # Output: [[0, 1, 2]]
  flatten_up_to([0, 1, 2], 0)  # Output: TypeError
  flatten_up_to([0, 1, 2], [0, 1, 2])  # Output: [0, 1, 2]
  ```

  Args:
    shallow_tree: a possibly pruned structure of input_tree.
    input_tree: an arbitrarily nested structure or a scalar object.
      Note, numpy arrays are considered scalars.
    check_types: bool. If True, check that each node in shallow_tree has the
      same type as the corresponding node in input_tree.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A Python list, the partially flattened version of `input_tree` according to
    the structure of `shallow_tree`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  is_seq = is_sequence_or_composite if expand_composites else is_sequence
  assert_shallow_structure(shallow_tree,
                           input_tree,
                           check_types=check_types,
                           expand_composites=expand_composites)
  # Discard paths returned by _yield_flat_up_to.
  return [v for _, v in _yield_flat_up_to(shallow_tree, input_tree, is_seq)]


def flatten_with_tuple_paths_up_to(shallow_tree,
                                   input_tree,
                                   check_types=True,
                                   expand_composites=False):
  """Flattens `input_tree` up to `shallow_tree`.

  Any further depth in structure in `input_tree` is retained as elements in the
  partially flattened output.

  Returns a list of (path, value) pairs, where value a leaf node in the
  flattened tree, and path is the tuple path of that leaf in input_tree.

  If `shallow_tree` and `input_tree` are not sequences, this returns a
  single-element list: `[((), input_tree)]`.

  Use Case:

  Sometimes we may wish to partially flatten a nested sequence, retaining some
  of the nested structure. We achieve this by specifying a shallow structure,
  `shallow_tree`, we wish to flatten up to.

  The input, `input_tree`, can be thought of as having the same structure layout
  as `shallow_tree`, but with leaf nodes that are themselves tree structures.

  Examples:

  ```python
  input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
  shallow_tree = [[True, True], [False, True]]

  flattened_input_tree = flatten_with_tuple_paths_up_to(shallow_tree,
                                                        input_tree)
  flattened_shallow_tree = flatten_with_tuple_paths_up_to(shallow_tree,
                                                          shallow_tree)

  # Output is:
  # [((0, 0), [2, 2]),
  #  ((0, 1), [3, 3]),
  #  ((1, 0), [4, 9]),
  #  ((1, 1), [5, 5])]
  #
  # [((0, 0), True),
  #  ((0, 1), True),
  #  ((1, 0), False),
  #  ((1, 1), True)]
  ```

  ```python
  input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
  shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]

  input_tree_flattened_as_shallow_tree = flatten_up_to(shallow_tree, input_tree)
  input_tree_flattened = flatten(input_tree)

  # Output is:
  # [((0, 0), ('a', 1)),
  #  ((0, 1, 0), ('b', 2)),
  #  ((0, 1, 1, 0), ('c', 3)),
  #  ((0, 1, 1, 1), ('d', 4))]
  # ['a', 1, 'b', 2, 'c', 3, 'd', 4]
  ```

  Non-Sequence Edge Cases:

  ```python
  flatten_with_tuple_paths_up_to(0, 0)  # Output: [(), 0]

  flatten_with_tuple_paths_up_to(0, [0, 1, 2])  # Output: [(), [0, 1, 2]]

  flatten_with_tuple_paths_up_to([0, 1, 2], 0)  # Output: TypeError

  flatten_with_tuple_paths_up_to([0, 1, 2], [0, 1, 2])
  # Output: [((0,) 0), ((1,), 1), ((2,), 2)]
  ```

  Args:
    shallow_tree: a possibly pruned structure of input_tree.
    input_tree: an arbitrarily nested structure or a scalar object.
      Note, numpy arrays are considered scalars.
    check_types: bool. If True, check that each node in shallow_tree has the
      same type as the corresponding node in input_tree.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A Python list, the partially flattened version of `input_tree` according to
    the structure of `shallow_tree`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  is_seq = is_sequence_or_composite if expand_composites else is_sequence
  assert_shallow_structure(shallow_tree,
                           input_tree,
                           check_types=check_types,
                           expand_composites=expand_composites)
  return list(_yield_flat_up_to(shallow_tree, input_tree, is_seq))


def map_structure_up_to(shallow_tree, func, *inputs, **kwargs):
  """Applies a function or op to a number of partially flattened inputs.

  The `inputs` are flattened up to `shallow_tree` before being mapped.

  Use Case:

  Sometimes we wish to apply a function to a partially flattened
  sequence (for example when the function itself takes sequence inputs). We
  achieve this by specifying a shallow structure, `shallow_tree` we wish to
  flatten up to.

  The `inputs`, can be thought of as having the same structure layout as
  `shallow_tree`, but with leaf nodes that are themselves tree structures.

  This function therefore will return something with the same base structure as
  `shallow_tree`.

  Examples:

  ```python
  shallow_tree = [None, None]
  inp_val = [1, 2, 3]
  out = map_structure_up_to(shallow_tree, lambda x: 2 * x, inp_val)

  # Output is: [2, 4]
  ```

  ```python
  ab_tuple = collections.namedtuple("ab_tuple", "a, b")
  op_tuple = collections.namedtuple("op_tuple", "add, mul")
  inp_val = ab_tuple(a=2, b=3)
  inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
  out = map_structure_up_to(inp_val, lambda val, ops: (val + ops.add) * ops.mul,
                            inp_val, inp_ops)

  # Output is: ab_tuple(a=6, b=15)
  ```

  ```python
  data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
  name_list = ['evens', ['odds', 'primes']]
  out = map_structure_up_to(
      name_list,
      lambda name, sec: "first_{}_{}".format(len(sec), name),
      name_list, data_list)

  # Output is: ['first_4_evens', ['first_5_odds', 'first_3_primes']]
  ```

  Args:
    shallow_tree: a shallow tree, common to all the inputs.
    func: callable which will be applied to each input individually.
    *inputs: arbitrarily nested combination of objects that are compatible with
        shallow_tree. The function `func` is applied to corresponding
        partially flattened elements of each input, so the function must support
        arity of `len(inputs)`.
    **kwargs: kwargs to feed to func(). Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.
      `map_structure(func, [1], (1,))` raises a `TypeError` exception). To allow
      this set this argument to `False`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.

  Returns:
    result of repeatedly applying `func`, with the same structure layout as
    `shallow_tree`.
  """
  return map_structure_with_tuple_paths_up_to(
      shallow_tree,
      lambda _, *values: func(*values),  # Discards the path arg.
      *inputs,
      **kwargs)


def map_structure_with_tuple_paths_up_to(shallow_tree, func, *inputs, **kwargs):
  """Applies a function or op to a number of partially flattened inputs.

  Like map_structure_up_to(), except that the 'func' argument takes a path
  tuple as its first argument, followed by the corresponding values from
  *inputs.

  Example:

  ```python
  lowercase = {'a': 'a', 'b': ('b0', 'b1')}
  uppercase = {'a': 'A', 'b': ('B0', 'B1')}

  def print_path_and_values(path, *values):
    print("path: {}, values: {}".format(path, values))

  shallow_tree = {'a': None}
  map_structure_with_tuple_paths_up_to(shallow_tree,
                                       print_path_and_values,
                                       lowercase,
                                       uppercase)
  path: ('a',), values: ('a', 'A')
  path: ('b', 0), values: ('b0', 'B0')
  path: ('b', 1), values: ('b1', 'B1')

  shallow_tree = {'b': None}
  map_structure_with_tuple_paths_up_to(shallow_tree,
                                       print_path_and_values,
                                       lowercase,
                                       uppercase,
                                       check_types=False)
  path: ('b', 1), values: (('bo', 'b1'), ('B0', 'B1'))

  shallow_tree = {'a': None, 'b': {1: None}}
  map_structure_with_tuple_paths_up_to(shallow_tree,
                                       print_path_and_values,
                                       lowercase,
                                       uppercase,
                                       check_types=False)
  path: ('a',), values: ('a', 'A')
  path: ('b', 1), values: ('b1', B1')
  ```

  Args:
    shallow_tree: a shallow tree, common to all the inputs.
    func: callable that takes args (path, inputs_0_value, ... , inputs_N_value),
      where path is a tuple path to a leaf node in shallow_tree, and
      inputs_i_value is the corresponding value from inputs[i].
    *inputs: nested structures that are all structurally compatible with
        shallow_tree.
    **kwargs: kwargs to feed to func(). Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.
      `map_structure(func, [1], (1,))` raises a `TypeError` exception). To allow
      this set this argument to `False`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but one of `*inputs` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.

  Returns:
    Result of repeatedly applying `func`. Has the same structure layout as
    `shallow_tree`.
  """
  if not inputs:
    raise ValueError("Cannot map over no sequences")

  check_types = kwargs.pop("check_types", True)
  expand_composites = kwargs.pop("expand_composites", False)
  is_seq = is_sequence_or_composite if expand_composites else is_sequence

  for input_tree in inputs:
    assert_shallow_structure(
        shallow_tree,
        input_tree,
        check_types=check_types,
        expand_composites=expand_composites)

  # Flatten each input separately, apply the function to corresponding elements,
  # then repack based on the structure of the first input.
  flat_value_gen = (
      flatten_up_to(  # pylint: disable=g-complex-comprehension
          shallow_tree,
          input_tree,
          check_types,
          expand_composites=expand_composites) for input_tree in inputs)
  flat_path_gen = (
      path for path, _ in _yield_flat_up_to(shallow_tree, inputs[0], is_seq))
  results = [
      func(*args, **kwargs) for args in zip(flat_path_gen, *flat_value_gen)
  ]
  return pack_sequence_as(structure=shallow_tree, flat_sequence=results,
                          expand_composites=expand_composites)


def get_traverse_shallow_structure(traverse_fn, structure,
                                   expand_composites=False):
  """Generates a shallow structure from a `traverse_fn` and `structure`.

  `traverse_fn` must accept any possible subtree of `structure` and return
  a depth=1 structure containing `True` or `False` values, describing which
  of the top-level subtrees may be traversed.  It may also
  return scalar `True` or `False` "traversal is OK / not OK for all subtrees."

  Examples are available in the unit tests (nest_test.py).

  Args:
    traverse_fn: Function taking a substructure and returning either a scalar
      `bool` (whether to traverse that substructure or not) or a depth=1
      shallow structure of the same type, describing which parts of the
      substructure to traverse.
    structure: The structure to traverse.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A shallow structure containing python bools, which can be passed to
    `map_structure_up_to` and `flatten_up_to`.

  Raises:
    TypeError: if `traverse_fn` returns a sequence for a non-sequence input,
      or a structure with depth higher than 1 for a sequence input,
      or if any leaf values in the returned structure or scalar are not type
      `bool`.
  """
  is_seq = is_sequence_or_composite if expand_composites else is_sequence
  to_traverse = traverse_fn(structure)
  if not is_seq(structure):
    if not isinstance(to_traverse, bool):
      raise TypeError("traverse_fn returned structure: %s for non-structure: %s"
                      % (to_traverse, structure))
    return to_traverse
  level_traverse = []
  if isinstance(to_traverse, bool):
    if not to_traverse:
      # Do not traverse this substructure at all.  Exit early.
      return False
    else:
      # Traverse the entire substructure.
      for branch in _yield_value(structure):
        level_traverse.append(
            get_traverse_shallow_structure(traverse_fn, branch,
                                           expand_composites=expand_composites))
  elif not is_seq(to_traverse):
    raise TypeError("traverse_fn returned a non-bool scalar: %s for input: %s"
                    % (to_traverse, structure))
  else:
    # Traverse some subset of this substructure.
    assert_shallow_structure(to_traverse, structure,
                             expand_composites=expand_composites)
    for t, branch in zip(_yield_value(to_traverse),
                         _yield_value(structure)):
      if not isinstance(t, bool):
        raise TypeError(
            "traverse_fn didn't return a depth=1 structure of bools.  saw: %s "
            " for structure: %s" % (to_traverse, structure))
      if t:
        level_traverse.append(
            get_traverse_shallow_structure(traverse_fn, branch))
      else:
        level_traverse.append(False)
  return _sequence_like(structure, level_traverse)


def yield_flat_paths(nest, expand_composites=False):
  """Yields paths for some nested structure.

  Paths are lists of objects which can be str-converted, which may include
  integers or other types which are used as indices in a dict.

  The flat list will be in the corresponding order as if you called
  `nest.flatten` on the structure. This is handy for naming Tensors such
  the TF scope structure matches the tuple structure.

  E.g. if we have a tuple `value = Foo(a=3, b=Bar(c=23, d=42))`

  ```shell
  nest.flatten(value)
  [3, 23, 42]
  list(nest.yield_flat_paths(value))
  [('a',), ('b', 'c'), ('b', 'd')]
  ```

  ```shell
  list(nest.yield_flat_paths({'a': [3]}))
  [('a', 0)]
  list(nest.yield_flat_paths({'a': 3}))
  [('a',)]
  ```

  Args:
    nest: the value to produce a flattened paths list for.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Yields:
    Tuples containing index or key values which form the path to a specific
    leaf value in the nested structure.
  """
  is_seq = is_sequence_or_composite if expand_composites else is_sequence
  for k, _ in _yield_flat_up_to(nest, nest, is_seq):
    yield k


def flatten_with_joined_string_paths(structure, separator="/",
                                     expand_composites=False):
  """Returns a list of (string path, data element) tuples.

  The order of tuples produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each data element was located. See `nest.yield_flat_paths`
  for more information.

  Args:
    structure: the nested structure to flatten.
    separator: string to separate levels of hierarchy in the results, defaults
      to '/'.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A list of (string, data element) tuples.
  """
  flat_paths = yield_flat_paths(structure, expand_composites=expand_composites)
  def stringify_and_join(path_elements):
    return separator.join(str(path_element) for path_element in path_elements)

  flat_string_paths = (stringify_and_join(path) for path in flat_paths)
  return list(zip(flat_string_paths,
                  flatten(structure, expand_composites=expand_composites)))


def flatten_with_tuple_paths(structure, expand_composites=False):
  """Returns a list of `(tuple_path, leaf_element)` tuples.

  The order of pairs produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each data element was located. See `nest.yield_flat_paths`
  for more information about tuple paths.

  Args:
    structure: the nested structure to flatten.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A list of `(tuple_path, leaf_element)` tuples. Each `tuple_path` is a tuple
    of indices and/or dictionary keys that uniquely specify the path to
    `leaf_element` within `structure`.
  """
  return list(zip(yield_flat_paths(structure,
                                   expand_composites=expand_composites),
                  flatten(structure, expand_composites=expand_composites)))


def list_to_tuple(structure):
  """Replace all lists with tuples.

  The fork of nest that tf.data uses treats lists as single elements, while
  tf.nest treats them as structures to recurse into. Keras has chosen to adopt
  the latter convention, and must therefore deeply replace all lists with tuples
  before passing structures to Dataset.from_generator.

  Args:
    structure: A nested structure to be remapped.

  Returns:
    structure mapped to replace all lists with tuples.
  """
  def sequence_fn(instance, args):
    if isinstance(instance, list):
      return tuple(args)
    return _sequence_like(instance, args)

  return _pack_sequence_as(structure, flatten(structure), False,
                           sequence_fn=sequence_fn)


_pywrap_utils.RegisterType("Mapping", _collections_abc.Mapping)
_pywrap_utils.RegisterType("MutableMapping", _collections_abc.MutableMapping)
_pywrap_utils.RegisterType("Sequence", _collections_abc.Sequence)
_pywrap_utils.RegisterType("MappingView", _collections_abc.MappingView)
_pywrap_utils.RegisterType("ObjectProxy", _wrapt.ObjectProxy)
