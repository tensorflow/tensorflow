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
"""A `traverse` visitor for processing documentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect


class DocGeneratorVisitor(object):
  """A visitor that generates docs for a python object when __call__ed."""

  def __init__(self, root_name=''):
    """Make a visitor.

    As this visitor is starting its traversal at a module or class, it will not
    be told the name of that object during traversal. `root_name` is the name it
    should use for that object, effectively prefixing all names with
    "root_name.".

    Args:
      root_name: The name of the root module/class.
    """
    self.set_root_name(root_name)
    self._index = {}
    self._tree = {}
    self._reverse_index = None
    self._duplicates = None
    self._duplicate_of = None

  def set_root_name(self, root_name):
    """Sets the root name for subsequent __call__s."""
    self._root_name = root_name or ''
    self._prefix = (root_name + '.') if root_name else ''

  @property
  def index(self):
    """A map from fully qualified names to objects to be documented.

    The index is filled when the visitor is passed to `traverse`.

    Returns:
      The index filled by traversal.
    """
    return self._index

  @property
  def tree(self):
    """A map from fully qualified names to all its child names for traversal.

    The full name to member names map is filled when the visitor is passed to
    `traverse`.

    Returns:
      The full name to member name map filled by traversal.
    """
    return self._tree

  @property
  def reverse_index(self):
    """A map from `id(object)` to the preferred fully qualified name.

    This map only contains non-primitive objects (no numbers or strings) present
    in `index` (for primitive objects, `id()` doesn't quite do the right thing).

    It is computed when it, `duplicate_of`, or `duplicates` are first accessed.

    Returns:
      The `id(object)` to full name map.
    """
    self._maybe_find_duplicates()
    return self._reverse_index

  @property
  def duplicate_of(self):
    """A map from duplicate full names to a preferred fully qualified name.

    This map only contains names that are not themself a preferred name.

    It is computed when it, `reverse_index`, or `duplicates` are first accessed.

    Returns:
      The map from duplicate name to preferred name.
    """
    self._maybe_find_duplicates()
    return self._duplicate_of

  @property
  def duplicates(self):
    """A map from preferred full names to a list of all names for this symbol.

    This function returns a map from preferred (master) name for a symbol to a
    lexicographically sorted list of all aliases for that name (incl. the master
    name). Symbols without duplicate names do not appear in this map.

    It is computed when it, `reverse_index`, or `duplicate_of` are first
    accessed.

    Returns:
      The map from master name to list of all duplicate names.
    """
    self._maybe_find_duplicates()
    return self._duplicates

  def _add_prefix(self, name):
    """Adds the root name to a name."""
    return self._prefix + name if name else self._root_name

  def __call__(self, parent_name, parent, children):
    """Visitor interface, see `tensorflow/tools/common:traverse` for details.

    This method is called for each symbol found in a traversal using
    `tensorflow/tools/common:traverse`. It should not be called directly in
    user code.

    Args:
      parent_name: The fully qualified name of a symbol found during traversal.
      parent: The Python object referenced by `parent_name`.
      children: A list of `(name, py_object)` pairs enumerating, in alphabetical
        order, the children (as determined by `tf_inspect.getmembers`) of
          `parent`. `name` is the local name of `py_object` in `parent`.

    Raises:
      RuntimeError: If this visitor is called with a `parent` that is not a
        class or module.
    """
    parent_name = self._add_prefix(parent_name)
    self._index[parent_name] = parent
    self._tree[parent_name] = []

    if not (tf_inspect.ismodule(parent) or tf_inspect.isclass(parent)):
      raise RuntimeError('Unexpected type in visitor -- %s: %r' % (parent_name,
                                                                   parent))

    for i, (name, child) in enumerate(list(children)):
      # Don't document __metaclass__
      if name in ['__metaclass__']:
        del children[i]
        continue

      full_name = '.'.join([parent_name, name]) if parent_name else name
      self._index[full_name] = child
      self._tree[parent_name].append(name)

  def _score_name(self, name):
    """Return a tuple of scores indicating how to sort for the best name.

    This function is meant to be used as the `key` to the `sorted` function.

    This sorting in order:
      Prefers names refering to the defining class, over a subclass.
      Prefers names that are not in "contrib".
      prefers submodules to the root namespace.
      Prefers short names `tf.thing` over `tf.a.b.c.thing`
      Sorts lexicographically on name parts.

    Args:
      name: the full name to score, for example `tf.estimator.Estimator`

    Returns:
      A tuple of scores. When sorted the preferred name will have the lowest
      value.
    """
    parts = name.split('.')
    short_name = parts[-1]

    container = self._index['.'.join(parts[:-1])]

    defining_class_score = 1
    if tf_inspect.isclass(container):
      if short_name in container.__dict__:
        # prefer the defining class
        defining_class_score = -1

    contrib_score = -1
    if 'contrib' in parts:
      contrib_score = 1

    while parts:
      parts.pop()
      container = self._index['.'.join(parts)]
      if tf_inspect.ismodule(container):
        break
    module_length = len(parts)
    if len(parts) == 2:
      # `tf.submodule.thing` is better than `tf.thing`
      module_length_score = -1
    else:
      # shorter is better
      module_length_score = module_length

    return (defining_class_score, contrib_score, module_length_score, name)

  def _maybe_find_duplicates(self):
    """Compute data structures containing information about duplicates.

    Find duplicates in `index` and decide on one to be the "master" name.

    Computes a reverse_index mapping each object id to its master name.

    Also computes a map `duplicate_of` from aliases to their master name (the
    master name itself has no entry in this map), and a map `duplicates` from
    master names to a lexicographically sorted list of all aliases for that name
    (incl. the master name).

    All these are computed and set as fields if they haven't already.
    """
    if self._reverse_index is not None:
      return

    # Maps the id of a symbol to its fully qualified name. For symbols that have
    # several aliases, this map contains the first one found.
    # We use id(py_object) to get a hashable value for py_object. Note all
    # objects in _index are in memory at the same time so this is safe.
    reverse_index = {}

    # Make a preliminary duplicates map. For all sets of duplicate names, it
    # maps the first name found to a list of all duplicate names.
    raw_duplicates = {}
    for full_name, py_object in six.iteritems(self._index):
      # We cannot use the duplicate mechanism for some constants, since e.g.,
      # id(c1) == id(c2) with c1=1, c2=1. This is unproblematic since constants
      # have no usable docstring and won't be documented automatically.
      if (py_object is not None and
          not isinstance(py_object, six.integer_types + six.string_types +
                         (six.binary_type, six.text_type, float, complex, bool))
          and py_object is not ()):  # pylint: disable=literal-comparison
        object_id = id(py_object)
        if object_id in reverse_index:
          master_name = reverse_index[object_id]
          if master_name in raw_duplicates:
            raw_duplicates[master_name].append(full_name)
          else:
            raw_duplicates[master_name] = [master_name, full_name]
        else:
          reverse_index[object_id] = full_name
    # Decide on master names, rewire duplicates and make a duplicate_of map
    # mapping all non-master duplicates to the master name. The master symbol
    # does not have an entry in this map.
    duplicate_of = {}
    # Duplicates maps the main symbols to the set of all duplicates of that
    # symbol (incl. itself).
    duplicates = {}
    for names in raw_duplicates.values():
      names = sorted(names)
      master_name = (
          tf_export.get_canonical_name_for_symbol(self._index[names[0]])
          if names else None)
      if master_name:
        master_name = 'tf.%s' % master_name
      else:
        # Choose the master name with a lexical sort on the tuples returned by
        # by _score_name.
        master_name = min(names, key=self._score_name)
        print(names, master_name)

      duplicates[master_name] = names
      for name in names:
        if name != master_name:
          duplicate_of[name] = master_name

      # Set the reverse index to the canonical name.
      reverse_index[id(self._index[master_name])] = master_name

    self._duplicate_of = duplicate_of
    self._duplicates = duplicates
    self._reverse_index = reverse_index
