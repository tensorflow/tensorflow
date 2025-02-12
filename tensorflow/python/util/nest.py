# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Functions that work with structures.

A structure is either:

* one of the recognized Python collections, holding _nested structures_;
* a value of any other type, typically a TensorFlow data type like Tensor,
  Variable, or of compatible types such as int, float, ndarray, etc. these are
  commonly referred to as _atoms_ of the structure.

A structure of type `T` is a structure whose atomic items are of type `T`.
For example, a structure of `tf.Tensor` only contains `tf.Tensor` as its atoms.

Historically a _nested structure_ was called a _nested sequence_ in TensorFlow.
A nested structure is sometimes called a _nest_ or a _tree_, but the formal
name _nested structure_ is preferred.

Refer to [Nesting Data Structures]
(https://en.wikipedia.org/wiki/Nesting_(computing)#Data_structures).

The following collection types are recognized by `tf.nest` as nested
structures:

* `collections.abc.Sequence` (except `string` and `bytes`).
  This includes `list`, `tuple`, and `namedtuple`.
* `collections.abc.Mapping` (with sortable keys).
  This includes `dict` and `collections.OrderedDict`.
* `collections.abc.MappingView` (with sortable keys).
* [`attr.s` classes](https://www.attrs.org/).
* Classes (including
  [`dataclass`](https://docs.python.org/library/dataclasses.html))
  that implement the `__tf_flatten__` and `__tf_unflatten__` methods.
  See examples in
  [`nest_util.py`](https://github.com/tensorflow/tensorflow/blob/04869b4e63bfc03cb13627b3e1b879fdd0f69e34/tensorflow/python/util/nest_util.py#L97)

Any other values are considered **atoms**.  Not all collection types are
considered nested structures.  For example, the following types are
considered atoms:

* `set`; `{"a", "b"}` is an atom, while `["a", "b"]` is a nested structure.
* [`dataclass` classes](https://docs.python.org/library/dataclasses.html) that
don't implement the custom flattening/unflattening methods mentioned above.
* `tf.Tensor`.
* `numpy.array`.

`tf.nest.is_nested` checks whether an object is a nested structure or an atom.
For example:

  >>> tf.nest.is_nested("1234")
  False
  >>> tf.nest.is_nested([1, 3, [4, 5]])
  True
  >>> tf.nest.is_nested(((7, 8), (5, 6)))
  True
  >>> tf.nest.is_nested([])
  True
  >>> tf.nest.is_nested({"a": 1, "b": 2})
  True
  >>> tf.nest.is_nested({"a": 1, "b": 2}.keys())
  True
  >>> tf.nest.is_nested({"a": 1, "b": 2}.values())
  True
  >>> tf.nest.is_nested({"a": 1, "b": 2}.items())
  True
  >>> tf.nest.is_nested(set([1, 2]))
  False
  >>> ones = tf.ones([2, 3])
  >>> tf.nest.is_nested(ones)
  False

Note: A proper structure shall form a tree. The user shall ensure there is no
cyclic references within the items in the structure,
i.e., no references in the structure of the input of these functions
should be recursive. The behavior is undefined if there is a cycle.

API docstring: tensorflow.nest
"""

from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.tf_export import tf_export


STRUCTURES_HAVE_MISMATCHING_LENGTHS = (
    nest_util.STRUCTURES_HAVE_MISMATCHING_LENGTHS
)

STRUCTURES_HAVE_MISMATCHING_TYPES = nest_util.STRUCTURES_HAVE_MISMATCHING_TYPES

SHALLOW_TREE_HAS_INVALID_KEYS = nest_util.SHALLOW_TREE_HAS_INVALID_KEYS

INPUT_TREE_SMALLER_THAN_SHALLOW_TREE = (
    nest_util.INPUT_TREE_SMALLER_THAN_SHALLOW_TREE
)

IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ = (
    "If shallow structure is a sequence, input must also be a sequence. "
    "Input has type: {}."
)

is_namedtuple = nest_util.is_namedtuple
_is_namedtuple = nest_util.is_namedtuple
_is_attrs = _pywrap_utils.IsAttrs
_is_mapping = _pywrap_utils.IsMapping
same_namedtuples = nest_util.same_namedtuples


def _yield_value(iterable):
  return nest_util.yield_value(nest_util.Modality.CORE, iterable)


def _yield_sorted_items(iterable):
  return nest_util.yield_sorted_items(nest_util.Modality.CORE, iterable)


@tf_export("__internal__.nest.is_mapping", v1=[])
def is_mapping(obj):
  """Returns a true if its input is a collections.Mapping."""
  return _is_mapping(obj)


# TODO(b/225045380): Move to a "leaf" library to use in trace_type.
@tf_export("__internal__.nest.is_attrs", v1=[])
def is_attrs(obj):
  """Returns a true if its input is an instance of an attr.s decorated class."""
  return _is_attrs(obj)


@tf_export("__internal__.nest.sequence_like", v1=[])
def _sequence_like(instance, args):
  """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`,
        `collections.OrderedDict`, or `composite_tensor.Composite_Tensor`
        or `type_spec.TypeSpec`.
    args: items to be converted to the `instance` type.

  Returns:
    `args` with the type of `instance`.
  """
  return nest_util.sequence_like(instance, args)


_is_nested_or_composite = _pywrap_utils.IsNestedOrComposite


@tf_export("nest.is_nested")
def is_nested(seq):
  """Returns true if its input is a nested structure.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a nested structure.

  Args:
    seq: the value to test.

  Returns:
    True if the input is a nested structure.
  """
  return nest_util.is_nested(nest_util.Modality.CORE, seq)


def is_nested_or_composite(seq):
  """Returns true if its input is a nested structure or a composite.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a nested structure.

  Args:
    seq: the value to test.

  Returns:
    True if the input is a nested structure or a composite.
  """
  return _is_nested_or_composite(seq)


def is_sequence_or_composite(seq):
  return _is_nested_or_composite(seq)


@tf_export("nest.flatten")
def flatten(structure, expand_composites=False):
  """Returns a flat list from a given structure.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
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
    structure: an atom or a nested structure. Note, numpy arrays are considered
      atoms and are not flattened.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A Python list, the flattened version of the input.

  Raises:
    TypeError: The nest is or contains a dict with non-sortable keys.
  """
  return nest_util.flatten(
      nest_util.Modality.CORE, structure, expand_composites
  )


@tf_export("nest.assert_same_structure")
def assert_same_structure(nest1, nest2, check_types=True,
                          expand_composites=False):
  """Asserts that two structures are nested in the same way.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure.

  Note the method does not check the types of atoms inside the structures.

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

  Args:
    nest1: an atom or a nested structure.
    nest2: an atom or a nested structure.
    check_types: if `True` (default) types of structures are checked as well,
      including the keys of dictionaries. If set to `False`, for example a list
      and a tuple of objects will look the same if they have the same size. Note
      that namedtuples with identical name and fields are always considered to
      have the same shallow structure. Two types will also be considered the
      same if they are both list subtypes (which allows "list" and
      "_ListWrapper" from trackable dependency tracking to compare equal).
      `check_types=True` only checks type of sub-structures. The types of atoms
      are not checked.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Raises:
    ValueError: If the two structures do not have the same number of atoms or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures. Only possible if `check_types` is `True`.
  """
  nest_util.assert_same_structure(
      nest_util.Modality.CORE, nest1, nest2, check_types, expand_composites
  )


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
  return _pywrap_nest.FlattenDictItems(dictionary)


@tf_export("nest.pack_sequence_as")
def pack_sequence_as(structure, flat_sequence, expand_composites=False):
  """Returns a given flattened sequence packed into a given structure.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
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
      atom counts.
    TypeError: `structure` is or contains a dict with non-sortable keys.
  """
  return nest_util.pack_sequence_as(
      nest_util.Modality.CORE, structure, flat_sequence, expand_composites
  )


@tf_export("nest.map_structure")
def map_structure(func, *structure, **kwargs):
  """Creates a new structure by applying `func` to each atom in `structure`.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
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

  Args:
    func: A callable that accepts as many arguments as there are structures.
    *structure: atom or nested structure.
    **kwargs: Valid keyword args are:
      * `check_types`: If set to `True` (default) the types of iterables within
        the structures have to be same (e.g. `map_structure(func, [1], (1,))`
        raises a `TypeError` exception). To allow this set this argument to
        `False`. Note that namedtuples with identical name and fields are always
        considered to have the same shallow structure.
      * `expand_composites`: If set to `True`, then composite tensors such as
        `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
        component tensors.  If `False` (the default), then composite tensors are
        not expanded.

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
  return nest_util.map_structure(
      nest_util.Modality.CORE, func, *structure, **kwargs
  )


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

  return nest_util.map_structure_up_to(
      nest_util.Modality.CORE, structure[0], wrapper_func, *structure, **kwargs
  )


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
  return nest_util.map_structure_up_to(
      nest_util.Modality.CORE, structure[0], func, *structure, **kwargs
  )


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
  nest_util.assert_shallow_structure(
      nest_util.Modality.CORE,
      shallow_tree,
      input_tree,
      check_types,
      expand_composites,
  )


@tf_export("__internal__.nest.flatten_up_to", v1=[])
def flatten_up_to(shallow_tree, input_tree, check_types=True,
                  expand_composites=False):
  """Flattens `input_tree` up to `shallow_tree`.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
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

  Edge Cases for atoms:

  ```python
  flatten_up_to(0, 0)  # Output: [0]
  flatten_up_to(0, [0, 1, 2])  # Output: [[0, 1, 2]]
  flatten_up_to([0, 1, 2], 0)  # Output: TypeError
  flatten_up_to([0, 1, 2], [0, 1, 2])  # Output: [0, 1, 2]
  ```

  Args:
    shallow_tree: a possibly pruned structure of input_tree.
    input_tree: an atom or a nested structure.
      Note, numpy arrays are considered atoms.
    check_types: bool. If True, check that each node in shallow_tree has the
      same type as the corresponding node in input_tree.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

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
  return nest_util.flatten_up_to(
      nest_util.Modality.CORE,
      shallow_tree,
      input_tree,
      check_types,
      expand_composites,
  )


def flatten_with_tuple_paths_up_to(shallow_tree,
                                   input_tree,
                                   check_types=True,
                                   expand_composites=False):
  """Flattens `input_tree` up to `shallow_tree`.

  Any further depth in structure in `input_tree` is retained as structures in
  the partially flattened output.

  Returns a list of (path, value) pairs, where value a leaf node in the
  flattened tree, and path is the tuple path of that leaf in input_tree.

  If `shallow_tree` and `input_tree` are not sequences, this returns a
  single-item list: `[((), input_tree)]`.

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
    input_tree: an atom or a nested structure.
      Note, numpy arrays are considered atoms.
    check_types: bool. If True, check that each node in shallow_tree has the
      same type as the corresponding node in input_tree.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

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
  is_nested_fn = _is_nested_or_composite if expand_composites else is_nested
  assert_shallow_structure(shallow_tree,
                           input_tree,
                           check_types=check_types,
                           expand_composites=expand_composites)
  return list(
      nest_util.yield_flat_up_to(
          nest_util.Modality.CORE, shallow_tree, input_tree, is_nested_fn
      )
  )


@tf_export("__internal__.nest.map_structure_up_to", v1=[])
def map_structure_up_to(shallow_tree, func, *inputs, **kwargs):
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
    shallow_tree: a shallow structure, common to all the inputs.
    func: callable which will be applied to each input individually.
    *inputs: structures that are compatible with shallow_tree. The function
        `func` is applied to corresponding structures due to partial flattening
        of each input, so the function must support arity of `len(inputs)`.
    **kwargs: kwargs to feed to func(). Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.
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
  return nest_util.map_structure_up_to(
      nest_util.Modality.CORE,
      shallow_tree,
      lambda _, *values: func(*values),  # Discards the path arg.
      *inputs,
      **kwargs,
  )


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
    shallow_tree: a shallow structure, common to all the inputs.
    func: callable that takes args (path, inputs_0_value, ... , inputs_N_value),
      where path is a tuple path to an atom in shallow_tree, and inputs_i_value
      is the corresponding value from inputs[i].
    *inputs: structures that are all structurally compatible with shallow_tree.
    **kwargs: kwargs to feed to func(). Special kwarg `check_types` is not
      passed to func, but instead determines whether the types of iterables
      within the structures have to be same (e.g. `map_structure(func, [1],
      (1,))` raises a `TypeError` exception). To allow this set this argument to
      `False`.

  Raises:
    TypeError: If `shallow_tree` is a nested structure but one of `*inputs` is
      not.
    TypeError: If the structure types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the structure lengths of `shallow_tree` are different from
      `input_tree`.

  Returns:
    Result of repeatedly applying `func`. Has the same structure layout as
    `shallow_tree`.
  """
  return nest_util.map_structure_up_to(
      nest_util.Modality.CORE, shallow_tree, func, *inputs, **kwargs
  )


@tf_export("__internal__.nest.get_traverse_shallow_structure", v1=[])
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
    TypeError: if `traverse_fn` returns a nested structure for an atom input.
      or a structure with depth higher than 1 for a nested structure input,
      or if any leaf values in the returned structure or scalar are not type
      `bool`.
  """
  is_nested_fn = _is_nested_or_composite if expand_composites else is_nested
  to_traverse = traverse_fn(structure)
  if not is_nested_fn(structure):
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
      for branch in nest_util.yield_value(nest_util.Modality.CORE, structure):
        level_traverse.append(
            get_traverse_shallow_structure(traverse_fn, branch,
                                           expand_composites=expand_composites))
  elif not is_nested_fn(to_traverse):
    raise TypeError("traverse_fn returned a non-bool scalar: %s for input: %s"
                    % (to_traverse, structure))
  else:
    # Traverse some subset of this substructure.
    assert_shallow_structure(to_traverse, structure,
                             expand_composites=expand_composites)
    for t, branch in zip(
        nest_util.yield_value(nest_util.Modality.CORE, to_traverse),
        nest_util.yield_value(nest_util.Modality.CORE, structure),
    ):
      if not isinstance(t, bool):
        raise TypeError(
            "traverse_fn didn't return a depth=1 structure of bools.  saw: %s "
            " for structure: %s" % (to_traverse, structure))
      if t:
        level_traverse.append(
            get_traverse_shallow_structure(traverse_fn, branch))
      else:
        level_traverse.append(False)
  return nest_util.sequence_like(structure, level_traverse)


@tf_export("__internal__.nest.yield_flat_paths", v1=[])
def yield_flat_paths(nest, expand_composites=False):
  """Yields paths for some nested structure.

  Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
  for the definition of a structure.

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
  is_nested_fn = _is_nested_or_composite if expand_composites else is_nested
  for k, _ in nest_util.yield_flat_up_to(
      nest_util.Modality.CORE, nest, nest, is_nested_fn
  ):
    yield k


def flatten_with_joined_string_paths(structure, separator="/",
                                     expand_composites=False):
  """Returns a list of (string path, atom) tuples.

  The order of tuples produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each atom was located. See `nest.yield_flat_paths`
  for more information.

  Args:
    structure: the nested structure to flatten.
    separator: string to separate levels of hierarchy in the results, defaults
      to '/'.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A list of (string, atom) tuples.
  """
  flat_paths = yield_flat_paths(structure, expand_composites=expand_composites)
  def stringify_and_join(path_elements):
    return separator.join(str(path_element) for path_element in path_elements)

  flat_string_paths = (stringify_and_join(path) for path in flat_paths)
  return list(zip(flat_string_paths,
                  flatten(structure, expand_composites=expand_composites)))


def flatten_with_tuple_paths(structure, expand_composites=False):
  """Returns a list of `(tuple_path, atom)` tuples.

  The order of pairs produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each atom was located. See `nest.yield_flat_paths`
  for more information about tuple paths.

  Args:
    structure: the nested structure to flatten.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A list of `(tuple_path, atom)` tuples. Each `tuple_path` is a tuple
    of indices and/or dictionary keys that uniquely specify the path to
    `atom` within `structure`.
  """
  return list(zip(yield_flat_paths(structure,
                                   expand_composites=expand_composites),
                  flatten(structure, expand_composites=expand_composites)))


@tf_export("__internal__.nest.list_to_tuple", v1=[])
def list_to_tuple(structure):
  """Replace all lists with tuples.

  The fork of nest that tf.data uses treats lists as atoms, while
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
    return nest_util.sequence_like(instance, args)

  return nest_util.pack_sequence_as(
      nest_util.Modality.CORE,
      structure,
      flatten(structure),
      False,
      sequence_fn=sequence_fn,
  )
