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
"""Tracing utilities used by SavedModel."""

from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun


def trace_save_and_restore(obj):
  """Traces `Trackable` serialize- and restore-from-tensors functions.

  Args:
    obj: A `Trackable` object.

  Returns:
    A concrete Function.
  """
  legacy_name = saveable_compat.get_saveable_name(obj)

  obj_save_fn = obj._serialize_to_tensors  # pylint: disable=protected-access
  obj_restore_fn = obj._restore_from_tensors  # pylint: disable=protected-access

  if isinstance(obj_save_fn, defun.ConcreteFunction):
    concrete_save = obj_save_fn
  else:
    @def_function.function
    def save_fn():
      tensor_dict = obj_save_fn()
      if legacy_name:
        # If there is a legacy decorator, append the name to the keys.
        return {f"{legacy_name}{key}": value
                for key, value in tensor_dict.items()}
      return tensor_dict

    concrete_save = save_fn.get_concrete_function()

  if isinstance(obj_restore_fn, defun.ConcreteFunction):
    concrete_restore = obj_restore_fn
  else:
    @def_function.function
    def restore_fn(restored_tensors):
      if legacy_name:
        # Do the opposite operation of save_fn()
        restored_tensors = {key[len(legacy_name):]: value
                            for key, value in restored_tensors.items()}
      obj_restore_fn(restored_tensors)

    concrete_restore = restore_fn.get_concrete_function(
        concrete_save.structured_outputs)

  return concrete_save, concrete_restore
