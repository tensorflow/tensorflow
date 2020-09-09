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
"""Core conversion logic, serves as main point of access."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import sys
import unittest

from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager import function
from tensorflow.python.util import tf_inspect


_ALLOWLIST_CACHE = cache.UnboundInstanceCache()


def _is_of_known_loaded_module(f, module_name):
  mod = sys.modules.get(module_name, None)
  if mod is None:
    return False
  if any(v is not None for v in mod.__dict__.values() if f is v):
    return True
  return False


def _is_known_loaded_type(f, module_name, entity_name):
  """Tests whether the function or method is an instance of a known type."""
  if (module_name not in sys.modules or
      not hasattr(sys.modules[module_name], entity_name)):
    return False
  type_entity = getattr(sys.modules[module_name], entity_name)
  if isinstance(f, type_entity):
    # The method if of this type. Example:
    #
    # o = ClassType()
    # function(o.method)()
    return True
  # Note: inspect is required here, to avoid unpacking tf.function decorators.
  if inspect.ismethod(f):
    # The the unbound method if of this type. Example:
    #
    # class ClassType:
    #   @function
    #   def method(self):
    #     ...
    # o = ClassType()
    # o.method()
    if isinstance(f.__func__, type_entity):
      return True
  return False


def is_unsupported(o):
  """Checks whether an entity is supported by AutoGraph at all."""

  # TODO(b/122265385): Remove this bypass.
  if (_is_known_loaded_type(o, 'wrapt', 'FunctionWrapper') or
      _is_known_loaded_type(o, 'wrapt', 'BoundFunctionWrapper')):
    logging.warn(
        '{} appears to be decorated by wrapt, which is not yet supported'
        ' by AutoGraph. The function will run as-is.'
        ' You may still apply AutoGraph before the wrapt decorator.'.format(o))
    logging.log(2, 'Permanently allowed: %s: wrapt decorated', o)
    return True

  if _is_known_loaded_type(o, 'functools', '_lru_cache_wrapper'):
    logging.log(2, 'Permanently allowed: %s: lru_cache', o)
    return True

  # Constructors are permanently allowed.
  # TODO(mdan): Toggle as experimental feature instead.
  # TODO(b/124016764): Remove this limitation.
  if inspect_utils.isconstructor(o):
    logging.log(2, 'Permanently allowed: %s: constructor', o)
    return True

  # Other built-in modules are permanently allowed.
  # TODO(mdan): Figure out how to do this consistently for all stdlib modules.
  if any(
      _is_of_known_loaded_module(o, m)
      for m in ('collections', 'pdb', 'copy', 'inspect', 're')):
    logging.log(2, 'Permanently allowed: %s: part of builtin module', o)
    return True

  # Custom ops and kernels are also permanently allowed.
  # See tensorflow.framework.load_library.
  if (hasattr(o, '__module__') and
      hasattr(o.__module__, '_IS_TENSORFLOW_PLUGIN')):
    logging.log(2, 'Permanently allowed: %s: TensorFlow plugin', o)
    return True

  return False


# TODO(mdan): allow_namedtuple_subclass should be hardcoded to True.
def is_allowlisted(
    o, check_call_override=True, allow_namedtuple_subclass=False):
  """Checks whether an entity is allowed for use in graph mode.

  Examples of allowed entities include all members of the tensorflow
  package.

  Args:
    o: A Python entity.
    check_call_override: Reserved for internal use. When set to `False`, it
      disables the rule according to which classes are allowed if their
      __call__ method is allowed.
    allow_namedtuple_subclass: Reserved for internal use. When `True`,
      namedtuple subclasses are not allowed.

  Returns:
    Boolean
  """
  # TODO(b/120224672): Fix this.
  if isinstance(o, functools.partial):
    # tf_inspect.getmodule(functools.partial(...)) otherwise returns None since
    # functools.partial objects do not have a __module__ attribute.
    m = functools
  else:
    m = tf_inspect.getmodule(o)

  # Examples of callables that lack a __module__ property include builtins.
  if hasattr(m, '__name__'):
    for rule in config.CONVERSION_RULES:
      action = rule.get_action(m)
      if action == config.Action.CONVERT:
        logging.log(2, 'Not allowed: %s: %s', o, rule)
        return False
      elif action == config.Action.DO_NOT_CONVERT:
        logging.log(2, 'Allowlisted: %s: %s', o, rule)
        return True

  # The check for __code__ below is because isgeneratorfunction crashes
  # without one.
  if hasattr(o, '__code__') and tf_inspect.isgeneratorfunction(o):
    logging.warn(
        'Entity %s appears to be a generator function. It will not be converted'
        ' by AutoGraph.', o)
    logging.log(2, 'Allowlisted: %s: generator functions are not converted', o)
    return True

  if (check_call_override and not tf_inspect.isclass(o) and
      hasattr(o, '__call__')):
    # Callable objects: allowed if their __call__ method is.
    # The type check avoids infinite recursion around the __call__ method
    # of function objects.
    if (type(o) != type(o.__call__)) and is_allowlisted(o.__call__):  # pylint: disable=unidiomatic-typecheck
      logging.log(2, 'Allowlisted: %s: object __call__ allowed', o)
      return True

  owner_class = None
  if tf_inspect.ismethod(o):
    # Methods of allowed classes are also allowed, even if they are
    # bound via user subclasses.
    #
    # For example, suppose `tf.Foo` has a method called `bar`, and `baz` is
    # defined as below. `tf.Foo` is allowed. Then `baz.bar` is also
    # allowed.
    #
    #   class Custom(tf.Foo):
    #     pass
    #
    #   baz = Custom()
    #
    # For the example above, if `Custom` did overload `bar`, then it would no
    # longer be allowed.

    owner_class = inspect_utils.getmethodclass(o)
    if owner_class is function.TfMethodTarget:
      owner_class = o.__self__.target_class
    if owner_class is not None:
      if issubclass(owner_class, unittest.TestCase):
        logging.log(2, 'Allowlisted: %s: method of TestCase subclass', o)
        return True

      owner_class = inspect_utils.getdefiningclass(o, owner_class)
      if is_allowlisted(
          owner_class,
          check_call_override=False,
          allow_namedtuple_subclass=True):
        logging.log(2, 'Allowlisted: %s: owner is allowed %s', o,
                    owner_class)
        return True

  if inspect_utils.isnamedtuple(o):
    # Due to the way they're constructed, namedtuple types cannot be converted
    # because they don't expose source code. But we assume they are safe for
    # graph mode since they are just containers.
    if allow_namedtuple_subclass:
      if not any(inspect_utils.isnamedtuple(base) for base in o.__bases__):
        logging.log(2, 'Allowlisted: %s: named tuple', o)
        return True
    else:
      logging.log(2, 'Allowlisted: %s: named tuple or subclass', o)
      return True

  logging.log(2, 'Not allowed: %s: default rule', o)
  return False


def is_in_allowlist_cache(entity, options):
  try:
    return _ALLOWLIST_CACHE.has(entity, options)
  except TypeError:
    # Catch-all for entities that are unhashable or don't allow weakrefs.
    return False


def cache_allowlisted(entity, options):
  try:
    _ALLOWLIST_CACHE[entity][options] = True
  except TypeError:
    # Catch-all for entities that are unhashable or don't allow weakrefs.
    pass
