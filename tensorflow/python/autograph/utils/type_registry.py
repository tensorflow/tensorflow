# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Registry mechanism implementing the registry pattern for general use."""


class TypeRegistry(object):
  """Provides a type registry for the python registry pattern.

  Contains mappings between types and type specific objects, to implement the
  registry pattern.

  Some example uses of this would be to register different functions depending
  on the type of object.
  """

  def __init__(self):
    self._registry = {}

  def register(self, obj, value):
    """Registers a Python object within the registry.

    Args:
      obj: The object to add to the registry.
      value: The stored value for the 'obj' type.

    Raises:
      KeyError: If the same obj is used twice.
    """
    if obj in self._registry:
      raise KeyError(f"{type(obj)} has already been registered.")
    self._registry[obj] = value

  def lookup(self, obj):
    """Looks up 'obj'.

    Args:
      obj: The object to lookup within the registry.

    Returns:
      Value for 'obj' in the registry if found.
    Raises:
      LookupError: if 'obj' has not been registered.
    """
    for registered in self._registry:
      if isinstance(
          obj, registered
      ):
        return self._registry[registered]

    raise LookupError(f"{type(obj)} has not been registered.")
