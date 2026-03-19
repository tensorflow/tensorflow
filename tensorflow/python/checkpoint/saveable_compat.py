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

  Needed for migrating certain Trackables (see next paragraph) from the legacy
  `_gather_saveables_for_checkpoint` to the new `_serialize_to_tensors`
  function.

  This decorator should be used if the SaveableObject generates tensors with
  different names from the name that is passed to the factory.

  Example migration:

  *Before*

  ```
  class MyTrackable(Trackable):
    def _gather_saveables_for_checkpoint(self):
      return {"key": _MySaveable}

  class _MySaveable(SaveableObject):
    def __init__(self, name):
      specs = [
          SaveSpec(tensor1, "", name + "-1")
          SaveSpec(tensor2, "", name + "-2")
      ]
      super().__init__(None, specs, name)
  ```

  *After*

  ```
  @legacy_saveable_name("key")
  class MyTrackable(Trackable):

    def _serialize_to_tensors(self):
      return {"key-1": tensor1, "key-2": tensor2}
  ```

  Args:
    name: String name of the SaveableObject factory (the key returned in the
       `_gather_saveables_for_checkpoint` function)

  Returns:
    A decorator.
  """
  def decorator(cls_or_obj):
    setattr(cls_or_obj, _LEGACY_SAVEABLE_NAME, name)
    return cls_or_obj
  return decorator


def get_saveable_name(cls_or_obj):
  return getattr(cls_or_obj, _LEGACY_SAVEABLE_NAME, None)


_FORCE_CHECKPOINT_CONVERSION = False


def force_checkpoint_conversion(value=True):
  """Forces checkpoint to use the new implementation.

  The new checkpoint implementation is changing the saved metadata slightly,
  and therefore may break forward compatibility in newly saved checkpoints. This
  means:

    - Previous versions of TensorFlow may not be able to load new checkpoints.
    - Backwards compatibility is unchanged: Old checkpoints can still be loaded.

  TensorFlow guarantees 3 weeks of forward compatibility, so this flag will be
  removed in the future weeks, after which checkpoint conversion will happen by
  default.

  **What happens when this flag is enabled?**

  The checkpoint will be saved with different metadata, meaning that previous
  versions of TensorFlow (<=2.10) will not be able to load this checkpoint.

  Args:
    value: Boolean value, whether or not to force checkpoint conversion to the
      new implementation.
  """
  # TODO(kathywu): Add definite date for flag removal.
  global _FORCE_CHECKPOINT_CONVERSION
  _FORCE_CHECKPOINT_CONVERSION = value


def force_checkpoint_conversion_enabled():
  return _FORCE_CHECKPOINT_CONVERSION


class CheckpointConversionError(Exception):
  pass
