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
"""Registry for custom TypeSpecs."""

import re

from tensorflow.python.types import internal


_TYPE_SPEC_TO_NAME = {}
_NAME_TO_TYPE_SPEC = {}

# Regular expression for valid TypeSpec names.
_REGISTERED_NAME_RE = re.compile(r"^(\w+\.)+\w+$")


# TODO(b/173744905) tf_export this as "tf.register_type_spec".  (And add a
# usage example to the docstring, once the API is public.)
#
# TODO(b/173744905) Update this decorator to apply to ExtensionType rather than
# TypeSpec (once we do refactoring to move to_components/from_components from
# TypeSpec to ExtensionType).
def register(name):
  """Decorator used to register a globally unique name for a TypeSpec subclass.

  Args:
    name: The name of the type spec.  Must be globally unique.  Must have the
      form `"{project_name}.{type_name}"`.  E.g. `"my_project.MyTypeSpec"`.

  Returns:
    A class decorator that registers the decorated class with the given name.
  """
  if not isinstance(name, str):
    raise TypeError("Expected `name` to be a string; got %r" % (name,))
  if not _REGISTERED_NAME_RE.match(name):
    raise ValueError(
        "Registered name must have the form '{project_name}.{type_name}' "
        "(e.g. 'my_project.MyTypeSpec'); got %r." % name)

  def decorator_fn(cls):
    if not (isinstance(cls, type) and issubclass(cls, internal.TypeSpec)):
      raise TypeError("Expected `cls` to be a TypeSpec; got %r" % (cls,))
    if cls in _TYPE_SPEC_TO_NAME:
      raise ValueError("Class %s.%s has already been registered with name %s." %
                       (cls.__module__, cls.__name__, _TYPE_SPEC_TO_NAME[cls]))
    if name in _NAME_TO_TYPE_SPEC:
      raise ValueError("Name %s has already been registered for class %s.%s." %
                       (name, _NAME_TO_TYPE_SPEC[name].__module__,
                        _NAME_TO_TYPE_SPEC[name].__name__))
    _TYPE_SPEC_TO_NAME[cls] = name
    _NAME_TO_TYPE_SPEC[name] = cls
    return cls

  return decorator_fn


# TODO(edloper) tf_export this as "tf.get_type_spec_name" (or some similar name)
def get_name(cls):
  """Returns the registered name for TypeSpec `cls`."""
  if not (isinstance(cls, type) and issubclass(cls, internal.TypeSpec)):
    raise TypeError("Expected `cls` to be a TypeSpec; got %r" % (cls,))
  if cls not in _TYPE_SPEC_TO_NAME:
    raise ValueError("TypeSpec %s.%s has not been registered." %
                     (cls.__module__, cls.__name__))
  return _TYPE_SPEC_TO_NAME[cls]


# TODO(edloper) tf_export this as "tf.lookup_type_spec" (or some similar name)
def lookup(name):
  """Returns the TypeSpec that has been registered with name `name`."""
  if not isinstance(name, str):
    raise TypeError("Expected `name` to be a string; got %r" % (name,))
  if name not in _NAME_TO_TYPE_SPEC:
    raise ValueError("No TypeSpec has been registered with name %r" % (name,))
  return _NAME_TO_TYPE_SPEC[name]
