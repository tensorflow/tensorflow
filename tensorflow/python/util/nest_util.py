# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Utility methods for handling nests.

This module encapsulates different semantics of handling nests by the public
tf.nest APIs and internal tf.data APIs. The difference in semantics exists for
historic reasons and reconciliation would require a non-backwards compatible
change.

The implementation of the different semantics use a common utility to
avoid / minimize further divergence between the two APIs over time.
"""

import collections as _collections
import enum

import six as _six
import wrapt as _wrapt

from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc


_is_mapping_view = _pywrap_utils.IsMappingView
_is_attrs = _pywrap_utils.IsAttrs
_is_composite_tensor = _pywrap_utils.IsCompositeTensor
_is_type_spec = _pywrap_utils.IsTypeSpec
_is_mutable_mapping = _pywrap_utils.IsMutableMapping
_is_mapping = _pywrap_utils.IsMapping
_tf_data_is_nested = _pywrap_utils.IsNestedForData
_tf_data_flatten = _pywrap_utils.FlattenForData
_tf_core_is_nested = _pywrap_utils.IsNested
_is_nested_or_composite = _pywrap_utils.IsNestedOrComposite
# See the swig file (util.i) for documentation.
same_namedtuples = _pywrap_utils.SameNamedtuples


STRUCTURES_HAVE_MISMATCHING_TYPES = (
    "The two structures don't have the same sequence type. Input structure has "
    "type {input_type}, while shallow structure has type {shallow_type}."
)

STRUCTURES_HAVE_MISMATCHING_LENGTHS = (
    "The two structures don't have the same sequence length. Input "
    "structure has length {input_length}, while shallow structure has length "
    "{shallow_length}."
)

INPUT_TREE_SMALLER_THAN_SHALLOW_TREE = (
    "The input_tree has fewer items than the shallow_tree. Input structure "
    "has length {input_size}, while shallow structure has length "
    "{shallow_size}."
)

SHALLOW_TREE_HAS_INVALID_KEYS = (
    "The shallow_tree's keys are not a subset of the input_tree's keys. The "
    "shallow_tree has the following keys that are not in the input_tree: {}."
)


class Modality(enum.Enum):
  """Modality/semantic used for treating nested structures.

  - Modality.CORE follows tensorflow_core/tf.nest semantics.

    The following collection types are recognized by `tf.nest` as nested
    structures:

    * `collections.abc.Sequence` (except `string` and `bytes`).
      This includes `list`, `tuple`, and `namedtuple`.
    * `collections.abc.Mapping` (with sortable keys).
      This includes `dict` and `collections.OrderedDict`.
    * `collections.abc.MappingView` (with sortable keys).
    * [`attr.s` classes](https://www.attrs.org/).

    Any other values are considered **atoms**.  Not all collection types are
    considered nested structures.  For example, the following types are
    considered atoms:

    * `set`; `{"a", "b"}` is an atom, while `["a", "b"]` is a nested structure.
    * [`dataclass` classes](https://docs.python.org/library/dataclasses.html)
    * `tf.Tensor`
    * `numpy.array`

  - Modality.DATA follows tf.data's nest semantics.

  This modality makes two changes:
  1. It removes support for lists as a level of nesting in nested structures.
  2. It adds support for `SparseTensorValue` as an atomic element.

  The motivation for this change is twofold:

  1. It seems more natural for lists to be treated (e.g. in Dataset
  constructors)
    as tensors, rather than lists of (lists of...) tensors.
  2. This is needed because `SparseTensorValue` is implemented as a `namedtuple`
    that would normally be flattened and we want to be able to create sparse
    tensor from `SparseTensorValue's similarly to creating tensors from numpy
    arrays.
  """

  CORE = "CORE"
  DATA = "DATA"


class _DotString(object):
  __slots__ = []

  def __str__(self):
    return "."

  def __repr__(self):
    return "."


_DOT = _DotString()


def is_nested(modality, structure):
  """Returns true if its input is a nested structure.

  For Modality.CORE refer to
  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a nested structure.

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    structure: the value to test.

  Returns:
    True if the input is a nested structure.
  """
  if modality == Modality.CORE:
    return _tf_core_is_nested(structure)
  elif modality == Modality.DATA:
    return _tf_data_is_nested(structure)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


# TODO(b/225045380): Move to a "leaf" library to use in trace_type.
def is_namedtuple(instance, strict=False):
  """Returns True iff `instance` is a `namedtuple`.

  Args:
    instance: An instance of a Python object.
    strict: If True, `instance` is considered to be a `namedtuple` only if it is
      a "plain" namedtuple. For instance, a class inheriting from a `namedtuple`
      will be considered to be a `namedtuple` iff `strict=False`.

  Returns:
    True if `instance` is a `namedtuple`.
  """
  return _pywrap_utils.IsNamedtuple(instance, strict)


def sequence_like(instance, args):
  """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`,
      `collections.OrderedDict`, or `composite_tensor.Composite_Tensor` or
      `type_spec.TypeSpec`.
    args: items to be converted to the `instance` type.

  Returns:
    `args` with the type of `instance`.
  """
  if _is_mutable_mapping(instance):
    # Pack dictionaries in a deterministic order by sorting the keys.
    # Notice this means that we ignore the original order of `OrderedDict`
    # instances. This is intentional, to avoid potential bugs caused by mixing
    # ordered and plain dicts (e.g., flattening a dict but using a
    # corresponding `OrderedDict` to pack it back).
    result = dict(zip(_tf_core_sorted(instance), args))
    instance_type = type(instance)
    if instance_type == _collections.defaultdict:
      d = _collections.defaultdict(instance.default_factory)
    else:
      d = instance_type()
    for key in instance:
      d[key] = result[key]
    return d
  elif _is_mapping(instance):
    result = dict(zip(_tf_core_sorted(instance), args))
    instance_type = type(instance)
    if not getattr(instance_type, "__supported_by_tf_nest__", False):
      tf_logging.log_first_n(
          tf_logging.WARN,
          "Mapping types may not work well with tf.nest. "
          "Prefer using MutableMapping for {}".format(instance_type),
          1,
      )
    try:
      return instance_type((key, result[key]) for key in instance)
    except TypeError as err:
      # pylint: disable=raise-missing-from
      raise TypeError(
          "Error creating an object of type {} like {}. Note that "
          "it must accept a single positional argument "
          "representing an iterable of key-value pairs, in "
          "addition to self. Cause: {}".format(type(instance), instance, err)
      )
  elif _is_mapping_view(instance):
    # We can't directly construct mapping views, so we create a list instead
    return list(args)
  elif is_namedtuple(instance) or _is_attrs(instance):
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
    return sequence_like(list(instance), args)
  elif isinstance(instance, _wrapt.ObjectProxy):
    # For object proxies, first create the underlying type and then re-wrap it
    # in the proxy type.
    return type(instance)(sequence_like(instance.__wrapped__, args))
  else:
    # Not a namedtuple
    return type(instance)(args)


def _get_attrs_items(obj):
  """Returns a list of (name, value) pairs from an attrs instance.

  TODO(b/268078256): check if this comment is valid, and if so, ensure it's
  handled in the function below.
  The list will be sorted by name.

  Args:
    obj: an object.

  Returns:
    A list of (attr_name, attr_value) pairs, sorted by attr_name.
  """
  attrs = getattr(obj.__class__, "__attrs_attrs__")
  attr_names = (a.name for a in attrs)
  return [(attr_name, getattr(obj, attr_name)) for attr_name in attr_names]


def _tf_core_sorted(dict_):
  """Returns a sorted list of the dict keys, with error if keys not sortable."""
  try:
    return sorted(dict_.keys())
  except TypeError:
    # pylint: disable=raise-missing-from
    raise TypeError("nest only supports dicts with sortable keys.")


def _tf_data_sorted(dict_):
  """Returns a sorted list of the dict keys, with error if keys not sortable."""
  try:
    return sorted(list(dict_))
  except TypeError as e:
    # pylint: disable=raise-missing-from
    raise TypeError(
        f"nest only supports dicts with sortable keys. Error: {e.message}"
    )


def yield_value(modality, iterable):
  """Yield elements of `iterable` in a deterministic order.

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    iterable: an iterable.

  Yields:
    The iterable elements in a deterministic order.
  """
  if modality == Modality.CORE:
    yield from _tf_core_yield_value(iterable)
  elif modality == Modality.DATA:
    yield from _tf_data_yield_value(iterable)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


def _tf_core_yield_value(iterable):
  for _, v in _tf_core_yield_sorted_items(iterable):
    yield v


def yield_sorted_items(modality, iterable):
  if modality == Modality.CORE:
    return _tf_core_yield_sorted_items(iterable)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


def _tf_core_yield_sorted_items(iterable):
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
    for key in _tf_core_sorted(iterable):
      yield key, iterable[key]
  elif _is_attrs(iterable):
    for item in _get_attrs_items(iterable):
      yield item
  elif is_namedtuple(iterable):
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


def _tf_data_yield_value(iterable):
  """Yield elements of `iterable` in a deterministic order.

  Args:
    iterable: an iterable.

  Yields:
    The iterable elements in a deterministic order.
  """
  # pylint: disable=protected-access
  if isinstance(iterable, _collections_abc.Mapping):
    # Iterate through dictionaries in a deterministic order by sorting the
    # keys. Notice this means that we ignore the original order of `OrderedDict`
    # instances. This is intentional, to avoid potential bugs caused by mixing
    # ordered and plain dicts (e.g., flattening a dict but using a
    # corresponding `OrderedDict` to pack it back).
    for key in _tf_data_sorted(iterable):
      yield iterable[key]
  # To avoid circular imports. sparse_tensor
  # depends on tensorflow/python/util/nest.py transitively, and if we try to
  # import sparse_tensor again, it results in a circular import. Instead, here
  # we check the class name instead of using `isinstance`.
  elif iterable.__class__.__name__ == "SparseTensorValue":
    yield iterable
  elif _is_attrs(iterable):
    for _, attr in _get_attrs_items(iterable):
      yield attr
  else:
    for value in iterable:
      yield value


def assert_same_structure(
    modality, nest1, nest2, check_types=True, expand_composites=False
):
  """Asserts that two structures are nested in the same way.

  For Modality.CORE refer to
  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure. Note the method does not check the types of
  atoms inside the structures.

  Examples:

  * These atom vs. atom comparisons will pass:

    >>> tf.nest.assert_same_structure(1.5, tf.Variable(1, tf.uint32))
    >>> tf.nest.assert_same_structure("abc", np.array([1, 2]))

  * These nested structure vs. nested structure comparisons will pass:

    >>> structure1 = (((1, 2), 3), 4, (5, 6))
    >>> structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    >>> structure3 = [(("a", "b"), "c"), "d", ["e", "f"]]
    >>> tf.nest.assert_same_structure(structure1, structure2)
    >>> tf.nest.assert_same_structure(structure1, structure3, check_types=False)

    >>> import collections
    >>> tf.nest.assert_same_structure(
    ...     collections.namedtuple("bar", "a b")(1, 2),
    ...     collections.namedtuple("foo", "a b")(2, 3),
    ...     check_types=False)

    >>> tf.nest.assert_same_structure(
    ...     collections.namedtuple("bar", "a b")(1, 2),
    ...     { "a": 1, "b": 2 },
    ...     check_types=False)

    >>> tf.nest.assert_same_structure(
    ...     { "a": 1, "b": 2, "c": 3 },
    ...     { "c": 6, "b": 5, "a": 4 })

    >>> ragged_tensor1 = tf.RaggedTensor.from_row_splits(
    ...       values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...       row_splits=[0, 4, 4, 7, 8, 8])
    >>> ragged_tensor2 = tf.RaggedTensor.from_row_splits(
    ...       values=[3, 1, 4],
    ...       row_splits=[0, 3])
    >>> tf.nest.assert_same_structure(
    ...       ragged_tensor1,
    ...       ragged_tensor2,
    ...       expand_composites=True)

  * These examples will raise exceptions:

    >>> tf.nest.assert_same_structure([0, 1], np.array([0, 1]))
    Traceback (most recent call last):
    ...
    ValueError: The two structures don't have the same nested structure

    >>> tf.nest.assert_same_structure(
    ...       collections.namedtuple('bar', 'a b')(1, 2),
    ...       collections.namedtuple('foo', 'a b')(2, 3))
    Traceback (most recent call last):
    ...
    TypeError: The two structures don't have the same nested structure

  For Modality.DATA, nested structures are treated differently than
  Modality.CORE. Please refer to class Modality's documentation above to read up
  on these differences.

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    nest1: an atom or a nested structure.
    nest2: an atom or a nested structure.
    check_types: - For Modality.CORE: if `True` (default) types of structures
      are checked as well, including the keys of dictionaries. If set to
      `False`, for example a list and a tuple of objects will look the same if
      they have the same size. Note that namedtuples with identical name and
      fields are always considered to have the same shallow structure. Two types
      will also be considered the same if they are both list subtypes (which
      allows "list" and "_ListWrapper" from trackable dependency tracking to
      compare equal). `check_types=True` only checks type of sub-structures. The
      types of atoms are not checked. - For Modality.DATA: if `True` (default)
      types of sequences should be same as well. For dictionary, "type" of
      dictionary is considered to include its keys. In other words, two
      dictionaries with different keys are considered to have a different
      "type". If set to `False`, two iterables are considered same as long as
      they yield the elements that have same structures.
    expand_composites: Arg only valid for Modality.CORE. If true, then composite
      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are
      expanded into their component tensors.

  Raises:
    ValueError: If the two structures do not have the same number of atoms or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures. Only possible if `check_types` is `True`.
  """
  if modality == Modality.CORE:
    _tf_core_assert_same_structure(nest1, nest2, check_types, expand_composites)
  elif modality == Modality.DATA:
    _tf_data_assert_same_structure(nest1, nest2, check_types)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


# pylint: disable=missing-function-docstring
def _tf_core_assert_same_structure(
    nest1, nest2, check_types=True, expand_composites=False
):
  # Convert to bool explicitly as otherwise pybind will not be able# to handle
  # type mismatch message correctly. See GitHub issue 42329 for details.
  check_types = bool(check_types)
  expand_composites = bool(expand_composites)
  try:
    _pywrap_utils.AssertSameStructure(
        nest1, nest2, check_types, expand_composites
    )
  except (ValueError, TypeError) as e:
    str1 = str(_tf_core_map_structure(lambda _: _DOT, nest1))
    str2 = str(_tf_core_map_structure(lambda _: _DOT, nest2))
    raise type(e)(
        "%s\nEntire first structure:\n%s\nEntire second structure:\n%s"
        % (str(e), str1, str2)
    )


def _tf_data_assert_same_structure(nest1, nest2, check_types=True):
  _pywrap_utils.AssertSameStructureForData(nest1, nest2, check_types)


def _tf_core_packed_nest_with_indices(
    structure, flat, index, is_nested_fn, sequence_fn=None
):
  """Helper function for pack_sequence_as.

  Args:
    structure: structure to mimic.
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.
    is_nested_fn: Function used to test if a value should be treated as a nested
      structure.
    sequence_fn: Function used to generate a new strcuture instance.

  Returns:
    The tuple (new_index, child), where:
      * new_index - the updated index into `flat` having processed `structure`.
      * packed - the subset of `flat` corresponding to `structure`,
                 having started at `index`, and packed into the same nested
                 format.

  Raises:
    ValueError: if `structure` contains more atoms than `flat`
      (assuming indexing starts from `index`).
  """
  packed = []
  sequence_fn = sequence_fn or sequence_like
  for s in _tf_core_yield_value(structure):
    if is_nested_fn(s):
      new_index, child = _tf_core_packed_nest_with_indices(
          s, flat, index, is_nested_fn, sequence_fn
      )
      packed.append(sequence_fn(s, child))
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed


def _tf_data_packed_nest_with_indices(structure, flat, index):
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
  for s in _tf_data_yield_value(structure):
    if _tf_data_is_nested(s):
      new_index, child = _tf_data_packed_nest_with_indices(s, flat, index)
      packed.append(sequence_like(s, child))  # pylint: disable=protected-access
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed


def flatten(modality, structure, expand_composites=False):
  """Flattens a nested structure.

  - For Modality.CORE: refer to
  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure.

  If the structure is an atom, then returns a single-item list: [structure].

  This is the inverse of the `nest.pack_sequence_as` method that takes in a
  flattened list and re-packs it into the nested structure.

  In the case of dict instances, the sequence consists of the values, sorted by
  key to ensure deterministic behavior. This is true also for OrderedDict
  instances: their sequence order is ignored, the sorting order of keys is used
  instead. The same convention is followed in `nest.pack_sequence_as`. This
  correctly repacks dicts and OrderedDicts after they have been flattened, and
  also allows flattening an OrderedDict and then repacking it back using a
  corresponding plain dict, or vice-versa. Dictionaries with non-sortable keys
  cannot be flattened.

  Users must not modify any collections used in nest while this function is
  running.

  Examples:

  1. Python dict (ordered by key):

    >>> dict = { "key3": "value3", "key1": "value1", "key2": "value2" }
    >>> tf.nest.flatten(dict)
    ['value1', 'value2', 'value3']

  2. For a nested python tuple:

    >>> tuple = ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)
    >>> tf.nest.flatten(tuple)
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

  3. For a nested dictionary of dictionaries:

    >>> dict = { "key3": {"c": (1.0, 2.0), "a": (3.0)},
    ... "key1": {"m": "val1", "g": "val2"} }
    >>> tf.nest.flatten(dict)
    ['val2', 'val1', 3.0, 1.0, 2.0]

  4. Numpy array (will not flatten):

    >>> array = np.array([[1, 2], [3, 4]])
    >>> tf.nest.flatten(array)
        [array([[1, 2],
                [3, 4]])]

  5. `tf.Tensor` (will not flatten):

    >>> tensor = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> tf.nest.flatten(tensor)
        [<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
          array([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]], dtype=float32)>]

  6. `tf.RaggedTensor`: This is a composite tensor thats representation consists
  of a flattened list of 'values' and a list of 'row_splits' which indicate how
  to chop up the flattened list into different rows. For more details on
  `tf.RaggedTensor`, please visit
  https://www.tensorflow.org/api_docs/python/tf/RaggedTensor.

  with `expand_composites=False`, we just return the RaggedTensor as is.

    >>> tensor = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2]])
    >>> tf.nest.flatten(tensor, expand_composites=False)
    [<tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2]]>]

  with `expand_composites=True`, we return the component Tensors that make up
  the RaggedTensor representation (the values and row_splits tensors)

    >>> tensor = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2]])
    >>> tf.nest.flatten(tensor, expand_composites=True)
    [<tf.Tensor: shape=(7,), dtype=int32, numpy=array([3, 1, 4, 1, 5, 9, 2],
                                                      dtype=int32)>,
     <tf.Tensor: shape=(4,), dtype=int64, numpy=array([0, 4, 4, 7])>]

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    structure: an atom or a nested structure. Note, numpy arrays are considered
      atoms and are not flattened.
    expand_composites: Arg valid for Modality.CORE only. If true, then composite
      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are
      expanded into their component tensors.

  Returns:
    A Python list, the flattened version of the input.

  Raises:
    TypeError: The nest is or contains a dict with non-sortable keys.
  """
  if modality == Modality.CORE:
    return _tf_core_flatten(structure, expand_composites)
  elif modality == Modality.DATA:
    return _tf_data_flatten(structure)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


def _tf_core_flatten(structure, expand_composites=False):
  """See comments for flatten() in tensorflow/python/util/nest.py."""
  if structure is None:
    return [None]
  expand_composites = bool(expand_composites)
  return _pywrap_utils.Flatten(structure, expand_composites)


def pack_sequence_as(
    modality, structure, flat_sequence, expand_composites, sequence_fn=None
):
  """Returns a given flattened sequence packed into a given structure.

  - For Modality.CORE: Refer to
  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure.

  If `structure` is an atom, `flat_sequence` must be a single-item list;
  in this case the return value is `flat_sequence[0]`.

  If `structure` is or contains a dict instance, the keys will be sorted to
  pack the flat sequence in deterministic order. This is true also for
  `OrderedDict` instances: their sequence order is ignored, the sorting order of
  keys is used instead. The same convention is followed in `flatten`.
  This correctly repacks dicts and `OrderedDict`s after they have been
  flattened, and also allows flattening an `OrderedDict` and then repacking it
  back using a corresponding plain dict, or vice-versa.
  Dictionaries with non-sortable keys cannot be flattened.

  Examples:

  1. Python dict:

    >>> structure = { "key3": "", "key1": "", "key2": "" }
    >>> flat_sequence = ["value1", "value2", "value3"]
    >>> tf.nest.pack_sequence_as(structure, flat_sequence)
    {'key3': 'value3', 'key1': 'value1', 'key2': 'value2'}

  2. For a nested python tuple:

    >>> structure = (('a','b'), ('c','d','e'), 'f')
    >>> flat_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> tf.nest.pack_sequence_as(structure, flat_sequence)
    ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)

  3. For a nested dictionary of dictionaries:

    >>> structure = { "key3": {"c": ('alpha', 'beta'), "a": ('gamma')},
    ...               "key1": {"e": "val1", "d": "val2"} }
    >>> flat_sequence = ['val2', 'val1', 3.0, 1.0, 2.0]
    >>> tf.nest.pack_sequence_as(structure, flat_sequence)
    {'key3': {'c': (1.0, 2.0), 'a': 3.0}, 'key1': {'e': 'val1', 'd': 'val2'}}

  4. Numpy array (considered a scalar):

    >>> structure = ['a']
    >>> flat_sequence = [np.array([[1, 2], [3, 4]])]
    >>> tf.nest.pack_sequence_as(structure, flat_sequence)
    [array([[1, 2],
           [3, 4]])]

  5. tf.Tensor (considered a scalar):

    >>> structure = ['a']
    >>> flat_sequence = [tf.constant([[1., 2., 3.], [4., 5., 6.]])]
    >>> tf.nest.pack_sequence_as(structure, flat_sequence)
    [<tf.Tensor: shape=(2, 3), dtype=float32,
     numpy= array([[1., 2., 3.], [4., 5., 6.]], dtype=float32)>]

  6. `tf.RaggedTensor`: This is a composite tensor thats representation consists
  of a flattened list of 'values' and a list of 'row_splits' which indicate how
  to chop up the flattened list into different rows. For more details on
  `tf.RaggedTensor`, please visit
  https://www.tensorflow.org/api_docs/python/tf/RaggedTensor.

  With `expand_composites=False`, we treat RaggedTensor as a scalar.

    >>> structure = { "foo": tf.ragged.constant([[1, 2], [3]]),
    ...               "bar": tf.constant([[5]]) }
    >>> flat_sequence = [ "one", "two" ]
    >>> tf.nest.pack_sequence_as(structure, flat_sequence,
    ... expand_composites=False)
    {'foo': 'two', 'bar': 'one'}

  With `expand_composites=True`, we expect that the flattened input contains
  the tensors making up the ragged tensor i.e. the values and row_splits
  tensors.

    >>> structure = { "foo": tf.ragged.constant([[1., 2.], [3.]]),
    ...               "bar": tf.constant([[5.]]) }
    >>> tensors = tf.nest.flatten(structure, expand_composites=True)
    >>> print(tensors)
    [<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[5.]],
     dtype=float32)>,
     <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1., 2., 3.],
     dtype=float32)>,
     <tf.Tensor: shape=(3,), dtype=int64, numpy=array([0, 2, 3])>]
    >>> verified_tensors = [tf.debugging.check_numerics(t, 'invalid tensor: ')
    ...                     if t.dtype==tf.float32 else t
    ...                     for t in tensors]
    >>> tf.nest.pack_sequence_as(structure, verified_tensors,
    ...                          expand_composites=True)
    {'foo': <tf.RaggedTensor [[1.0, 2.0], [3.0]]>,
     'bar': <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[5.]],
     dtype=float32)>}

  - For Modality.DATA:  If `structure` is a scalar, `flat_sequence` must be a
  single-element list;
  in this case the return value is `flat_sequence[0]`.

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    structure: - For Modality.CORE: Nested structure, whose structure is given
      by nested lists, tuples, and dicts. Note: numpy arrays and strings are
      considered scalars. - For Modality.DATA: tuple or list constructed of
      scalars and/or other tuples/lists, or a scalar.  Note: numpy arrays are
      considered scalars.
    flat_sequence: flat sequence to pack.
    expand_composites: Arg valid for Modality.CORE only. If true, then composite
      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are
      expanded into their component tensors.
    sequence_fn: Arg valid for Modality.CORE only.

  Returns:
    packed: `flat_sequence` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If `flat_sequence` and `structure` have different
      atom counts.
    TypeError: For Modality.CORE only. `structure` is or contains a dict with
    non-sortable keys.
  """
  if modality == Modality.CORE:
    return _tf_core_pack_sequence_as(
        structure, flat_sequence, expand_composites, sequence_fn
    )
  elif modality == Modality.DATA:
    return _tf_data_pack_sequence_as(structure, flat_sequence)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


def _tf_core_pack_sequence_as(
    structure, flat_sequence, expand_composites, sequence_fn=None
):
  """Implements sequence packing, with the option to alter the structure."""
  is_nested_fn = (
      _is_nested_or_composite if expand_composites else _tf_core_is_nested
  )
  sequence_fn = sequence_fn or sequence_like

  def truncate(value, length):
    value_str = str(value)
    return value_str[:length] + (value_str[length:] and "...")

  if not is_nested_fn(flat_sequence):
    raise TypeError(
        "Attempted to pack value:\n  {}\ninto a structure, but found "
        "incompatible type `{}` instead.".format(
            truncate(flat_sequence, 100), type(flat_sequence)
        )
    )

  if not is_nested_fn(structure):
    if len(flat_sequence) != 1:
      raise ValueError(
          "The target structure is of type `{}`\n  {}\nHowever the input "
          "is a sequence ({}) of length {}.\n  {}\nnest cannot "
          "guarantee that it is safe to map one to the other.".format(
              type(structure),
              truncate(structure, 100),
              type(flat_sequence),
              len(flat_sequence),
              truncate(flat_sequence, 100),
          )
      )
    return flat_sequence[0]

  try:
    final_index, packed = _tf_core_packed_nest_with_indices(
        structure, flat_sequence, 0, is_nested_fn, sequence_fn
    )
    if final_index < len(flat_sequence):
      raise IndexError
  except IndexError:
    flat_structure = _tf_core_flatten(
        structure, expand_composites=expand_composites
    )
    if len(flat_structure) != len(flat_sequence):
      # pylint: disable=raise-missing-from
      raise ValueError(
          "Could not pack sequence. Structure had %d atoms, but "
          "flat_sequence had %d items.  Structure: %s, flat_sequence: %s."
          % (len(flat_structure), len(flat_sequence), structure, flat_sequence)
      )
  return sequence_fn(structure, packed)


def _tf_data_pack_sequence_as(structure, flat_sequence):
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
  if not (_tf_data_is_nested(flat_sequence) or isinstance(flat_sequence, list)):
    raise TypeError(
        "Argument `flat_sequence` must be a sequence. Got "
        f"'{type(flat_sequence).__name__}'."
    )

  if not _tf_data_is_nested(structure):
    if len(flat_sequence) != 1:
      raise ValueError(
          "Argument `structure` is a scalar but "
          f"`len(flat_sequence)`={len(flat_sequence)} > 1"
      )
    return flat_sequence[0]

  flat_structure = _tf_data_flatten(structure)
  if len(flat_structure) != len(flat_sequence):
    raise ValueError(
        "Could not pack sequence. Argument `structure` had "
        f"{len(flat_structure)} elements, but argument `flat_sequence` had "
        f"{len(flat_sequence)} elements. Received structure: "
        f"{structure}, flat_sequence: {flat_sequence}."
    )

  _, packed = _tf_data_packed_nest_with_indices(structure, flat_sequence, 0)
  return sequence_like(structure, packed)  # pylint: disable=protected-access


def map_structure(modality, func, *structure, **kwargs):
  """Creates a new structure by applying `func` to each atom in `structure`.

  - For Modality.CORE: Refer to
  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure.

  Applies `func(x[0], x[1], ...)` where x[i] enumerates all atoms in
  `structure[i]`.  All items in `structure` must have the same arity,
  and the return value will contain results with the same structure layout.

  Examples:

  * A single Python dict:

  >>> a = {"hello": 24, "world": 76}
  >>> tf.nest.map_structure(lambda p: p * 2, a)
  {'hello': 48, 'world': 152}

  * Multiple Python dictionaries:

  >>> d1 = {"hello": 24, "world": 76}
  >>> d2 = {"hello": 36, "world": 14}
  >>> tf.nest.map_structure(lambda p1, p2: p1 + p2, d1, d2)
  {'hello': 60, 'world': 90}

  * A single Python list:

  >>> a = [24, 76, "ab"]
  >>> tf.nest.map_structure(lambda p: p * 2, a)
  [48, 152, 'abab']

  * Scalars:

  >>> tf.nest.map_structure(lambda x, y: x + y, 3, 4)
  7

  * Empty structures:

  >>> tf.nest.map_structure(lambda x: x + 1, ())
  ()

  * Check the types of iterables:

  >>> s1 = (((1, 2), 3), 4, (5, 6))
  >>> s1_list = [[[1, 2], 3], 4, [5, 6]]
  >>> tf.nest.map_structure(lambda x, y: None, s1, s1_list)
  Traceback (most recent call last):
  ...
  TypeError: The two structures don't have the same nested structure

  * Type check is set to False:

  >>> s1 = (((1, 2), 3), 4, (5, 6))
  >>> s1_list = [[[1, 2], 3], 4, [5, 6]]
  >>> tf.nest.map_structure(lambda x, y: None, s1, s1_list, check_types=False)
  (((None, None), None), None, (None, None))

  - For Modality.DATA: Applies `func(x[0], x[1], ...)` where x[i] is an entry in
  `structure[i]`.  All structures in `structure` must have the same arity,
  and the return value will contain the results in the same structure.

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    func: A callable that accepts as many arguments as there are structures.
    *structure: - For Modality.CORE: atom or nested structure. - For
      Modality.DATA: scalar, or tuple or list of constructed scalars and/or
      other tuples/lists, or scalars.  Note: numpy arrays are considered
      scalars.
    **kwargs: Valid keyword args are: * `check_types`: - For Modality.CORE: If
      set to `True` (default) the types of iterables within the structures have
      to be same (e.g. `map_structure(func, [1], (1,))` raises a `TypeError`
      exception). To allow this set this argument to `False`. Note that
      namedtuples with identical name and fields are always considered to have
      the same shallow structure. - For Modality.DATA: only valid keyword
      argument is `check_types`. If set to `True` (default) the types of
      iterables within the structures have to be same (e.g. `map_structure(func,
      [1], (1,))` raises a `TypeError` exception). To allow this set this
      argument to `False`. * `expand_composites`: Valid for Modality.CORE only.
      If set to `True`, then composite tensors such as `tf.sparse.SparseTensor`
      and `tf.RaggedTensor` are expanded into their component tensors.  If
      `False` (the default), then composite tensors are not expanded.

  Returns:
    A new structure with the same arity as `structure[0]`, whose atoms
    correspond to `func(x[0], x[1], ...)` where `x[i]` is the atom in the
    corresponding location in `structure[i]`. If there are different structure
    types and `check_types` is `False` the structure types of the first
    structure will be used.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    ValueError: If no structure is provided or if the structures do not match
      each other by type.
    ValueError: If wrong keyword arguments are provided.
  """
  if modality == Modality.CORE:
    return _tf_core_map_structure(func, *structure, **kwargs)
  elif modality == Modality.DATA:
    return _tf_data_map_structure(func, *structure, **kwargs)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


# pylint: disable=missing-function-docstring
def _tf_core_map_structure(func, *structure, **kwargs):
  if not callable(func):
    raise TypeError("func must be callable, got: %s" % func)

  if not structure:
    raise ValueError("Must provide at least one structure")

  check_types = kwargs.pop("check_types", True)
  expand_composites = kwargs.pop("expand_composites", False)

  if kwargs:
    raise ValueError(
        "Only valid keyword arguments are `check_types` and "
        "`expand_composites`, not: `%s`"
        % "`, `".join(kwargs.keys())
    )

  for other in structure[1:]:
    _tf_core_assert_same_structure(
        structure[0],
        other,
        check_types=check_types,
        expand_composites=expand_composites,
    )

  flat_structure = (_tf_core_flatten(s, expand_composites) for s in structure)
  entries = zip(*flat_structure)

  return _tf_core_pack_sequence_as(
      structure[0],
      [func(*x) for x in entries],
      expand_composites=expand_composites,
  )


# pylint: disable=missing-function-docstring
def _tf_data_map_structure(func, *structure, **check_types_dict):
  if not callable(func):
    raise TypeError(f"Argument `func` must be callable, got: {func}")

  if not structure:
    raise ValueError("Must provide at least one structure")

  if check_types_dict:
    if "check_types" not in check_types_dict or len(check_types_dict) > 1:
      raise ValueError(
          "Only valid keyword argument for `check_types_dict` is "
          f"'check_types'. Got {check_types_dict}."
      )
    check_types = check_types_dict["check_types"]
  else:
    check_types = True

  for other in structure[1:]:
    _tf_data_assert_same_structure(structure[0], other, check_types=check_types)

  flat_structure = (_tf_data_flatten(s) for s in structure)
  entries = zip(*flat_structure)

  return _tf_data_pack_sequence_as(structure[0], [func(*x) for x in entries])


def yield_flat_up_to(modality, shallow_tree, input_tree, is_nested_fn, path=()):
  """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

  - For Modality.CORE: See comments for _tf_core_yield_flat_up_to() below
  - For Modality.DATA: See comments for _tf_data_yield_flat_up_to() below

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    shallow_tree: Nested structure. Traverse no further than its leaf nodes.
    input_tree: Nested structure. Return the paths and values from this tree.
      Must have the same upper structure as shallow_tree.
    is_nested_fn: Arg valid for Modality.CORE only. Function used to test if a
      value should be treated as a nested structure.
    path: Arg valid for Modality.CORE only. Tuple. Optional argument, only used
      when recursing. The path from the root of the original shallow_tree, down
      to the root of the shallow_tree arg of this recursive call.

  Yields:
    Pairs of (path, value), where path the tuple path of a leaf node in
    shallow_tree, and value is the value of the corresponding node in
    input_tree.
  """
  if modality == Modality.CORE:
    yield from _tf_core_yield_flat_up_to(
        shallow_tree, input_tree, is_nested_fn, path
    )
  elif modality == Modality.DATA:
    yield from _tf_data_yield_flat_up_to(shallow_tree, input_tree)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


def _tf_core_yield_flat_up_to(shallow_tree, input_tree, is_nested_fn, path=()):
  """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

  Args:
    shallow_tree: Nested structure. Traverse no further than its leaf nodes.
    input_tree: Nested structure. Return the paths and values from this tree.
      Must have the same upper structure as shallow_tree.
    is_nested_fn: Function used to test if a value should be treated as a nested
      structure.
    path: Tuple. Optional argument, only used when recursing. The path from the
      root of the original shallow_tree, down to the root of the shallow_tree
      arg of this recursive call.

  Yields:
    Pairs of (path, value), where path the tuple path of a leaf node in
    shallow_tree, and value is the value of the corresponding node in
    input_tree.
  """
  if not is_nested_fn(shallow_tree):
    yield (path, input_tree)
  else:
    input_tree = dict(_tf_core_yield_sorted_items(input_tree))
    for (
        shallow_key,
        shallow_subtree,
    ) in _tf_core_yield_sorted_items(shallow_tree):
      subpath = path + (shallow_key,)
      input_subtree = input_tree[shallow_key]
      for leaf_path, leaf_value in _tf_core_yield_flat_up_to(
          shallow_subtree, input_subtree, is_nested_fn, path=subpath
      ):
        yield (leaf_path, leaf_value)


def _tf_data_yield_flat_up_to(shallow_tree, input_tree):
  """Yields elements `input_tree` partially flattened up to `shallow_tree`."""
  if _tf_data_is_nested(shallow_tree):
    for shallow_branch, input_branch in zip(
        _tf_data_yield_value(shallow_tree), _tf_data_yield_value(input_tree)
    ):
      for input_leaf in _tf_data_yield_flat_up_to(shallow_branch, input_branch):
        yield input_leaf
  else:
    yield input_tree


def assert_shallow_structure(
    modality,
    shallow_tree,
    input_tree,
    check_types=True,
    expand_composites=False,
):
  """Asserts that `shallow_tree` is a shallow structure of `input_tree`.

  This function tests if the `input_tree` structure can be created from
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
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    shallow_tree: an arbitrarily nested structure.
    input_tree: an arbitrarily nested structure.
    check_types: if `True` (default) the sequence types of `shallow_tree` and
      `input_tree` have to be the same. Note that even with check_types==True,
      this function will consider two different namedtuple classes with the same
      name and _fields attribute to be the same class.
    expand_composites: Valid for Modality.CORE only. If true, then composite
      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are
      expanded into their component tensors.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`. Only raised if `check_types` is `True`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  if modality == Modality.CORE:
    _tf_core_assert_shallow_structure(
        shallow_tree, input_tree, check_types, expand_composites
    )
  elif modality == Modality.DATA:
    _tf_data_assert_shallow_structure(shallow_tree, input_tree, check_types)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


# pylint: disable=missing-function-docstring
def _tf_core_assert_shallow_structure(
    shallow_tree, input_tree, check_types=True, expand_composites=False
):
  is_nested_fn = (
      _is_nested_or_composite if expand_composites else _tf_core_is_nested
  )
  if is_nested_fn(shallow_tree):
    if not is_nested_fn(input_tree):
      raise TypeError(
          "If shallow structure is a sequence, input must also be a sequence. "
          "Input has type: %s."
          % type(input_tree)
      )

    if isinstance(shallow_tree, _wrapt.ObjectProxy):
      shallow_type = type(shallow_tree.__wrapped__)
    else:
      shallow_type = type(shallow_tree)

    if check_types and not isinstance(input_tree, shallow_type):
      # Duck-typing means that nest should be fine with two different
      # namedtuples with identical name and fields.
      shallow_is_namedtuple = is_namedtuple(shallow_tree, False)
      input_is_namedtuple = is_namedtuple(input_tree, False)
      if shallow_is_namedtuple and input_is_namedtuple:
        if not same_namedtuples(shallow_tree, input_tree):
          raise TypeError(
              STRUCTURES_HAVE_MISMATCHING_TYPES.format(
                  input_type=type(input_tree), shallow_type=type(shallow_tree)
              )
          )

      elif isinstance(shallow_tree, list) and isinstance(input_tree, list):
        # List subclasses are considered the same,
        # e.g. python list vs. _ListWrapper.
        pass

      elif (
          _is_composite_tensor(shallow_tree) or _is_type_spec(shallow_tree)
      ) and (_is_composite_tensor(input_tree) or _is_type_spec(input_tree)):
        pass  # Compatibility will be checked below.

      elif not (
          isinstance(shallow_tree, _collections_abc.Mapping)
          and isinstance(input_tree, _collections_abc.Mapping)
      ):
        raise TypeError(
            STRUCTURES_HAVE_MISMATCHING_TYPES.format(
                input_type=type(input_tree), shallow_type=type(shallow_tree)
            )
        )

    if _is_composite_tensor(shallow_tree) or _is_composite_tensor(input_tree):
      if not (
          (_is_composite_tensor(input_tree) or _is_type_spec(input_tree))
          and (
              _is_composite_tensor(shallow_tree) or _is_type_spec(shallow_tree)
          )
      ):
        raise TypeError(
            STRUCTURES_HAVE_MISMATCHING_TYPES.format(
                input_type=type(input_tree), shallow_type=type(shallow_tree)
            )
        )
      # pylint: disable=protected-access
      type_spec_1 = (
          shallow_tree
          if _is_type_spec(shallow_tree)
          else shallow_tree._type_spec
      )._without_tensor_names()
      type_spec_2 = (
          input_tree if _is_type_spec(input_tree) else input_tree._type_spec
      )._without_tensor_names()
      # TODO(b/246356867): Replace the most_specific_common_supertype below
      # with get_structure.
      if hasattr(type_spec_1, "_get_structure") and hasattr(
          type_spec_2, "_get_structure"
      ):
        result = (
            type_spec_1._get_structure() == type_spec_2._get_structure() or None
        )
      else:
        result = type_spec_1.most_specific_common_supertype([type_spec_2])
      if result is None:
        raise ValueError(
            "Incompatible CompositeTensor TypeSpecs: %s vs. %s"
            % (type_spec_1, type_spec_2)
        )
      # pylint: enable=protected-access

    elif _is_type_spec(shallow_tree):
      if not _is_type_spec(input_tree):
        raise TypeError(
            "If shallow structure is a TypeSpec, input must also "
            "be a TypeSpec.  Input has type: %s."
            % type(input_tree)
        )
    else:
      if len(input_tree) != len(shallow_tree):
        raise ValueError(
            STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
                input_length=len(input_tree), shallow_length=len(shallow_tree)
            )
        )
      elif len(input_tree) < len(shallow_tree):
        raise ValueError(
            INPUT_TREE_SMALLER_THAN_SHALLOW_TREE.format(
                input_size=len(input_tree), shallow_size=len(shallow_tree)
            )
        )

    if isinstance(shallow_tree, _collections_abc.Mapping):
      absent_keys = set(shallow_tree) - set(input_tree)
      if absent_keys:
        raise ValueError(
            SHALLOW_TREE_HAS_INVALID_KEYS.format(sorted(absent_keys))
        )

    for shallow_branch, input_branch in zip(
        _tf_core_yield_value(shallow_tree),
        _tf_core_yield_value(input_tree),
    ):
      _tf_core_assert_shallow_structure(
          shallow_branch,
          input_branch,
          check_types=check_types,
          expand_composites=expand_composites,
      )


# pylint: disable=missing-function-docstring
def _tf_data_assert_shallow_structure(
    shallow_tree, input_tree, check_types=True
):
  if _tf_data_is_nested(shallow_tree):
    if not _tf_data_is_nested(input_tree):
      raise TypeError(
          "If shallow structure is a sequence, input must also be a sequence. "
          f"Input has type: '{type(input_tree).__name__}'."
      )

    if check_types and not isinstance(input_tree, type(shallow_tree)):
      raise TypeError(
          "The two structures don't have the same sequence type. Input "
          f"structure has type '{type(input_tree).__name__}', while shallow "
          f"structure has type '{type(shallow_tree).__name__}'."
      )

    if len(input_tree) != len(shallow_tree):
      raise ValueError(
          "The two structures don't have the same sequence length. Input "
          f"structure has length {len(input_tree)}, while shallow structure "
          f"has length {len(shallow_tree)}."
      )

    if check_types and isinstance(shallow_tree, _collections_abc.Mapping):
      if set(input_tree) != set(shallow_tree):
        raise ValueError(
            "The two structures don't have the same keys. Input "
            f"structure has keys {list(input_tree)}, while shallow structure "
            f"has keys {list(shallow_tree)}."
        )
      input_tree = sorted(input_tree.items())
      shallow_tree = sorted(shallow_tree.items())

    for shallow_branch, input_branch in zip(shallow_tree, input_tree):
      _tf_data_assert_shallow_structure(
          shallow_branch, input_branch, check_types=check_types
      )


def flatten_up_to(
    modality,
    shallow_tree,
    input_tree,
    check_types=True,
    expand_composites=False,
):
  # pylint: disable=g-doc-return-or-yield,g-doc-args
  """Flattens `input_tree` up to `shallow_tree`.

  - For Modality.CORE: refer to
  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure.

  Any further depth in structure in `input_tree` is retained as structures in
  the partially flatten output.

  If `shallow_tree` and `input_tree` are atoms, this returns a
  single-item list: `[input_tree]`.

  Use Case:

  Sometimes we may wish to partially flatten a structure, retaining some
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

  Edge Cases:

  ```python
  flatten_up_to(0, 0)  # Output: [0]
  flatten_up_to(0, [0, 1, 2])  # Output: [[0, 1, 2]]
  flatten_up_to([0, 1, 2], 0)  # Output: TypeError
  flatten_up_to([0, 1, 2], [0, 1, 2])  # Output: [0, 1, 2]

  ```

  Args:
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    shallow_tree: a possibly pruned structure of input_tree.
    input_tree: an atom or a nested structure. Note, numpy arrays are considered
      atoms.
    check_types: bool. If True, check that each node in shallow_tree has the
      same type as the corresponding node in input_tree.
    expand_composites: Arg valid for Modality.CORE only. If true, then composite
      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are
      expanded into their component tensors.

  Returns:
    A Python list, the partially flattened version of `input_tree` according to
    the structure of `shallow_tree`.

  Raises:
    TypeError: If `shallow_tree` is a nested structure but `input_tree` is not.
    TypeError: If the structure types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the structure lengths of `shallow_tree` are different from
      `input_tree`.
  """
  if modality == Modality.CORE:
    return _tf_core_flatten_up_to(
        shallow_tree, input_tree, check_types, expand_composites
    )
  elif modality == Modality.DATA:
    return _tf_data_flatten_up_to(shallow_tree, input_tree)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


def _tf_core_flatten_up_to(
    shallow_tree, input_tree, check_types=True, expand_composites=False
):
  is_nested_fn = (
      _is_nested_or_composite if expand_composites else _tf_core_is_nested
  )
  _tf_core_assert_shallow_structure(
      shallow_tree,
      input_tree,
      check_types=check_types,
      expand_composites=expand_composites,
  )
  # Discard paths returned by nest_util._tf_core_yield_flat_up_to.
  return [
      v
      for _, v in _tf_core_yield_flat_up_to(
          shallow_tree, input_tree, is_nested_fn
      )
  ]


def _tf_data_flatten_up_to(shallow_tree, input_tree):
  _tf_data_assert_shallow_structure(shallow_tree, input_tree)
  return list(_tf_data_yield_flat_up_to(shallow_tree, input_tree))


def map_structure_up_to(modality, shallow_tree, func, *inputs, **kwargs):
  """Applies a function or op to a number of partially flattened inputs.

  The `inputs` are flattened up to `shallow_tree` before being mapped.

  Use Case:

  Sometimes we wish to apply a function to a partially flattened
  structure (for example when the function itself takes structure inputs). We
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
    modality: enum value of supported modality [Modality.CORE or Modality.DATA]
    shallow_tree: a shallow structure, common to all the inputs.
    func: callable which will be applied to each input individually.
    *inputs: structures that are compatible with shallow_tree. The function
      `func` is applied to corresponding structures due to partial flattening of
      each input, so the function must support arity of `len(inputs)`.
    **kwargs: Arg valid for Modality.CORE only. kwargs to feed to func().
      Special kwarg `check_types` is not passed to func, but instead determines
      whether the types of iterables within the structures have to be same (e.g.
      `map_structure(func, [1], (1,))` raises a `TypeError` exception). To allow
      this set this argument to `False`.

  Raises:
    TypeError: If `shallow_tree` is a nested structure but `input_tree` is not.
    TypeError: If the structure types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the structure lengths of `shallow_tree` are different from
      `input_tree`.

  Returns:
    result of repeatedly applying `func`, with the same structure layout as
    `shallow_tree`.
  """
  if modality == Modality.CORE:
    return _tf_core_map_structure_with_tuple_paths_up_to(
        shallow_tree, func, *inputs, **kwargs
    )
  elif modality == Modality.DATA:
    return _tf_data_map_structure_up_to(shallow_tree, func, *inputs)
  else:
    raise ValueError(
        "Unknown modality used {} for nested structure".format(modality)
    )


def _tf_core_map_structure_with_tuple_paths_up_to(
    shallow_tree, func, *inputs, **kwargs
):
  """See comments for map_structure_with_tuple_paths_up_to() in tensorflow/python/util/nest.py."""
  if not inputs:
    raise ValueError("Cannot map over no sequences")

  check_types = kwargs.pop("check_types", True)
  expand_composites = kwargs.pop("expand_composites", False)
  is_nested_fn = (
      _is_nested_or_composite if expand_composites else _tf_core_is_nested
  )

  for input_tree in inputs:
    _tf_core_assert_shallow_structure(
        shallow_tree,
        input_tree,
        check_types=check_types,
        expand_composites=expand_composites,
    )

  # Flatten each input separately, apply the function to corresponding items,
  # then repack based on the structure of the first input.
  flat_value_gen = (
      _tf_core_flatten_up_to(  # pylint: disable=g-complex-comprehension
          shallow_tree,
          input_tree,
          check_types,
          expand_composites=expand_composites,
      )
      for input_tree in inputs
  )
  flat_path_gen = (
      path
      for path, _ in _tf_core_yield_flat_up_to(
          shallow_tree, inputs[0], is_nested_fn
      )
  )
  results = [
      func(*args, **kwargs) for args in zip(flat_path_gen, *flat_value_gen)
  ]
  return _tf_core_pack_sequence_as(
      structure=shallow_tree,
      flat_sequence=results,
      expand_composites=expand_composites,
  )


# pylint: disable=missing-function-docstring
def _tf_data_map_structure_up_to(shallow_tree, func, *inputs):
  if not inputs:
    raise ValueError(
        "Argument `inputs` is empty. Cannot map over no sequences."
    )
  for input_tree in inputs:
    _tf_data_assert_shallow_structure(shallow_tree, input_tree)

  # Flatten each input separately, apply the function to corresponding elements,
  # then repack based on the structure of the first input.
  all_flattened_up_to = (
      _tf_data_flatten_up_to(shallow_tree, input_tree) for input_tree in inputs
  )

  results = [func(*tensors) for tensors in zip(*all_flattened_up_to)]
  return _tf_data_pack_sequence_as(
      structure=shallow_tree, flat_sequence=results
  )
