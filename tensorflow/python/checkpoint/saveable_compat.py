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
"""Checkpoint compatibility functions with SaveableObject.

Compatibility methods to ensure that checkpoints are saved with the same
metadata attributes before/after the SaveableObject deprecation.
"""

_LEGACY_SAVEABLE_NAME = "_LEGACY_SAVEABLE_NAME"


def legacy_saveable_name(name):
  """Decorator to set the local name to use in the Checkpoint.

  Needed for migrating certain Trackables from the legacy
  `_gather_saveables_for_checkpoint` to the new `_serialize_to_tensors`
  function.

  This decorator should be used if the SaveableObject generates tensors with
  different names from the name that is passed to the factory.

  Args:
    name: String name of the SaveableObject factory (the key returned in the
       `_gather_saveables_for_checkpoint` function)

  Returns:
    A decorator.
  """
  def decorator(serialize_to_tensors_fn):
    setattr(serialize_to_tensors_fn, _LEGACY_SAVEABLE_NAME, name)
    return serialize_to_tensors_fn
  return decorator


def get_saveable_name(obj):
# pylint: disable=protected-access
  obj_serialize_fn = obj._serialize_to_tensors
  if hasattr(obj_serialize_fn, "__func__"):
    obj_serialize_fn = obj_serialize_fn.__func__
  return getattr(obj_serialize_fn, _LEGACY_SAVEABLE_NAME, None)
  # pylint: enable=protected-access
