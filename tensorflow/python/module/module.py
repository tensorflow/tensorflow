# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Modules encapsulate building stateful components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys

import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.training.checkpointable import tracking
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


class ModuleMetaclass(type):
  """Metaclass for `tf.Module`."""

  def __new__(mcs, name, bases, clsdict):
    for key, value in clsdict.items():
      if key in ("__init__", "name_scope"):
        continue

      elif tf_inspect.isfunction(value):
        if getattr(value, "_no_module_name_scope", False):
          # The function has been annotated to say that no autoscoping should
          # be applied, so do not patch it.
          continue
        clsdict[key] = with_name_scope(value)

      elif isinstance(value, property):
        clsdict[key] = property(
            value.fget if not value.fget else with_name_scope(value.fget),
            value.fset if not value.fset else with_name_scope(value.fset),
            value.fdel if not value.fdel else with_name_scope(value.fdel),
            doc=value.__doc__)

    return type.__new__(mcs, name, bases, clsdict)

  def __call__(cls, *args, **kwargs):
    # Call new such that we have an un-initialized module instance that we can
    # still reference even if there is an exception during __init__. This is
    # needed such that we can make sure the name_scope constructed in __init__
    # is closed even if there is an exception.
    module = cls.__new__(cls, *args, **kwargs)

    # Now attempt to initialize the object.
    try:
      module.__init__(*args, **kwargs)
    except:
      # We must explicitly catch so that in Python 2 sys.exc_info() is populated
      # before entering the finally block.
      raise

    finally:
      # The base Module constructor enters the modules name scope before
      # returning such that other functionality in the ctor happens within the
      # modules name scope.
      scope = getattr(module, "_ctor_name_scope", None)
      exc_info = sys.exc_info()
      if scope is None:
        if exc_info[0] is None:
          raise ValueError(
              "Constructing a tf.Module without calling the super constructor "
              "is not supported. Add the following as the first line in your "
              "__init__ method:\n\n"
              "super(%s, self).__init__()" % cls.__name__)
      else:
        scope.__exit__(*exc_info)
        del module._ctor_name_scope

    return module


def with_name_scope(unbound_method):
  """Patches the given method so it enters the modules name scope."""
  def enter_name_scope(self, *args, **kwargs):
    """Decorator that calls the given function in the module name scope.

    Args:
      self: Module instance.
      *args: Positional arguments to `unbound_method`.
      **kwargs: Keyword arguments to `unbound_method`.

    Returns:
      `with self.name_scope: return unbound_method(self, *args, **kwargs)`
    """
    try:
      module_name_scope = self.name_scope
    except AttributeError as exc_value_from:
      exc_value = AttributeError(
          "The super constructor must be called before any other methods in "
          "your constructor. If this is not possible then annotate all the "
          "methods called with `@no_module_name_scope`.")
      six.raise_from(exc_value, exc_value_from)

    with module_name_scope:
      # tf.Module enters the module name scope for all methods. To disable this
      # for a particular method annotate it with `@no_module_name_scope`.
      return unbound_method(self, *args, **kwargs)

  return tf_decorator.make_decorator(unbound_method, enter_name_scope)


@tf_export("experimental.Module")
class Module(six.with_metaclass(ModuleMetaclass, tracking.AutoCheckpointable)):
  """Base neural network module class.

  A module is a named container for `tf.Variable`s, other `tf.Module`s and
  functions which apply to user input. For example a dense layer in a neural
  network might be implemented as a `tf.Module`:

  >>> class Dense(tf.Module):
  ...   def __init__(self, in_features, output_features):
  ...     super(Dense, self).__init__()
  ...     self.w = tf.Variable(
  ...         tf.random_normal([input_features, output_features]), name='w')
  ...     self.b = tf.Variable(tf.zeros([output_features]), name='b')
  ...
  ...   def __call__(self, x):
  ...     x = tf.convert_to_tensor(x, name='x')
  ...     y = tf.matmul(x, self.w) + self.b
  ...     return tf.nn.relu(y)

  You can use the dense layer as you would expect:

  >>> d = Dense(input_features=64, output_features=10)
  >>> d(tf.ones([100, 64]))
  <tf.Tensor: ...>

  By subclassing `tf.Module` instead of `object` any variables created inside
  the module are automatically created within the modules name scope:

  >>> d.w.name
  "dense/w:0"

  In eager mode this is useful for debugging, and when used with `@tf.function`
  the use of name scopes gives operations (e.g. matmul) useful names as well.

  As well as automatic naming, the Dense module inherits methods for tracking
  its variables:

  >>> d.variables
  (<tf.Variable 'dense/b:0' ...>, <tf.Variable 'dense/w:0' ...>)
  """

  def __init__(self, name=None):
    if name is None:
      name = camel_to_snake(type(self).__name__)
    else:
      if not valid_identifier(name):
        raise ValueError(
            "%r is not a valid module name. Module names must be valid Python "
            "identifiers (e.g. a valid class name)." % name)

    self._name = name
    with ops.name_scope(name) as scope_name:
      self._scope_name = scope_name

    # Enter the name scope so subsequent code in the contructor (e.g. creating
    # submodules) happens inside the modules name scope. This is exited when
    # the subclass __init__ returns (this is implemented in ModuleMetaclass).
    self._ctor_name_scope = self.name_scope
    self._ctor_name_scope.__enter__()

  @property
  def name(self):
    """Returns the name of this module as passed or determined in the ctor.

    NOTE: This is not the same as the `self.name_scope.name` which includes
    parent module names.
    """
    return self._name

  @property
  def name_scope(self):
    """Returns a `tf.name_scope` instance for this class."""
    # TODO(tomhennigan) Memoize once name scopes are re-entrant.
    return ops.name_scope(self._scope_name)

  @property
  def variables(self):
    """Sequence of variables owned by this module and it's submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (depth first).
    """
    return tuple(self._flatten(predicate=_IS_VARIABLE))

  @property
  def owned_variables(self):
    """Sequence of variables that are attributes of the current module.

    See `variables` for a property which returns all variables from the current
    module and all it's submodules recursively.

    Returns:
      A sequence of variables which are attributes of the current module. Will
      yield variables inside nested structures (lists etc) but not in other
      modules.
    """
    return tuple(self._flatten(recursive=False, predicate=_IS_VARIABLE))

  @property
  def trainable_variables(self):
    """Sequence of variables owned by this module and it's submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (depth first).
    """
    return tuple(self._flatten(predicate=_IS_TRAINABLE_VARIABLE))

  @property
  def owned_trainable_variables(self):
    """Sequence of variables that are attributes of the current module.

    See `variables` for a property which returns all variables from the current
    module and all it's submodules recursively.

    Returns:
      A sequence of variables which are attributes of the current module. Will
      yield variables inside nested structures (lists etc) but not in other
      modules.
    """
    return tuple(
        self._flatten(recursive=False, predicate=_IS_TRAINABLE_VARIABLE))

  @property
  def owned_submodules(self):
    """Sequence of immediate child modules.

    Child modules are modules which are found as properties of the current
    module.

    >>> a = tf.experimental.Module()
    >>> b = tf.experimental.Module()
    >>> c = tf.experimental.Module()
    >>> a.b = b
    >>> b.c = c
    >>> assert list(a.owned_submodules) == [b]
    >>> assert list(b.owned_submodules) == [c]
    >>> assert list(c.owned_submodules) == []

    Returns:
      A sequence of all child modules.
    """
    return tuple(self._flatten(recursive=False, predicate=_IS_MODULE))

  @property
  def submodules(self):
    """Sequence of all sub-modules.

    Submodules are modules which are properties of this module, or found as
    properties of modules which are properties of this module (and so on).

    >>> a = tf.experimental.Module()
    >>> b = tf.experimental.Module()
    >>> c = tf.experimental.Module()
    >>> a.b = b
    >>> b.c = c
    >>> assert list(a.submodules) == [b, c]
    >>> assert list(b.submodules) == [c]
    >>> assert list(c.submodules) == []

    Returns:
      A sequence of all submodules.
    """
    return tuple(self._flatten(predicate=_IS_MODULE))

  def _flatten(self,
               recursive=True,
               predicate=None,
               attribute_traversal_key=None):
    """Flattened attribute values in sorted order by attribute name.

    Modules are flattened by first walking their attributes in name order.
    Each attribute value is then flattened to find leaf values. If flatten is
    to be applied `recursive`ly then if the leaf is a `Module` it will also be
    flattened to find leaves. Finally every leaf value is optionally tested
    against the given `predicate` and finally yielded.

    >>> class Foo(tf.experimental.Module):
    ...   def __init__(self):
    ...     super(Foo, self).__init__()
    ...     self.x = [tf.constant('a'), tf.constant('b')]
    ...     self.y = {'i': tf.constant('c'), 'j': tf.constant('d')}
    ...     self.z = tf.constant('e')
    ...
    ...   @property
    ...   def tensors(self):
    ...     return tuple(self._flatten(predicate=is_tensor))

    >>> foo = Foo()
    >>> foo.tensors
    (<tf.Tensor...'a'>, <tf.Tensor...'b'>, ...'c'>, ...'d'>, ...'e'>)

    `attribute_traversal_key` controls the order object properties are visited.
    If not set objects are visited in ascending order by name.

    Args:
      recursive: Whether to recurse into child modules or not.
      predicate: (Optional) If set then only values matching predicate are
        yielded. A value of `None` (the default) means no items will be
        filtered.
      attribute_traversal_key: (Optional) Method to rekey object attributes
        before they are sorted. Contract is the same as `key` argument to
        builtin `sorted` and only applies to object properties.

    Returns:
      Flat generator for leaves of the current module and optionally all
      submodules.
    """
    if predicate is None:
      predicate = lambda _: True

    return _flatten_module(
        self,
        recursive=recursive,
        predicate=predicate,
        attribute_traversal_key=attribute_traversal_key,
        seen=set())

  @classmethod
  def no_name_scope(cls, method):
    """Decorator to wrap a method, preventing automatic name scope wrapping.

    By default, any method on a module is considered as a forwards function, and
    so any variables / modules created by the method will be scoped as belonging
    to the module. In some cases this is undesirable, for example when
    implementing .clone() / .transpose(), as in those cases we want the new
    module to have the scope of wherever the .transpose() call is made. To
    allow this, decorate any methods with `no_module_name_scope`.

    This logic is tied to ModuleMetaclass.__new__, if anything is
    changed here corresponding changes will be needed there.

    Args:
      method: the method to wrap.

    Returns:
      The method, with a flag indicating no name scope wrapping should occur.
    """
    setattr(method, "_no_module_name_scope", True)
    return method

_IS_VARIABLE = lambda o: isinstance(o, variables.Variable)
_IS_TRAINABLE_VARIABLE = lambda o: (_IS_VARIABLE(o) and o.trainable)
_IS_MODULE = lambda o: isinstance(o, Module)
_CAMEL_TO_SNAKE_R = re.compile(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")
_VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_]([a-zA-Z0-9_])*$")


def valid_identifier(name):
  return bool(_VALID_IDENTIFIER.match(name))


def camel_to_snake(value):
  return _CAMEL_TO_SNAKE_R.sub(r"_\1", value).lower()


def _flatten_module(module, recursive, predicate, attribute_traversal_key,
                    seen):
  """Implementation of `flatten`."""
  if seen is None:
    seen = set([id(module)])

  module_dict = vars(module)
  submodules = []

  for key in sorted(module_dict, key=attribute_traversal_key):
    for leaf in nest.flatten(module_dict[key]):
      leaf_id = id(leaf)
      if leaf_id in seen:
        continue

      seen.add(leaf_id)
      if predicate(leaf):
        yield leaf

      if recursive and isinstance(leaf, Module):
        # Walk direct properties first then recurse.
        submodules.append(leaf)

  for submodule in submodules:
    subvalues = _flatten_module(
        submodule,
        recursive=recursive,
        predicate=predicate,
        attribute_traversal_key=attribute_traversal_key,
        seen=seen)

    for subvalue in subvalues:
      # Predicate is already tested for these values.
      yield subvalue
