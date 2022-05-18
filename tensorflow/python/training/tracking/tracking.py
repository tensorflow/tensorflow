# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Various types of Trackable objects."""


from tensorflow.python.eager import function_saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.trackable import resource


AutoTrackable = autotrackable.AutoTrackable
Asset = asset.Asset
ResourceTracker = resource.ResourceTracker
resource_tracker_scope = resource.resource_tracker_scope
CapturableResource = resource.CapturableResource
TrackableResource = resource.TrackableResource


def convert_to_trackable(obj, parent=None):
  """Converts `obj` to `Trackable`."""
  if isinstance(obj, base.Trackable):
    return obj
  obj = data_structures.wrap_or_unwrap(obj)
  if (tensor_util.is_tf_type(obj) and
      obj.dtype not in (dtypes.variant, dtypes.resource) and
      not resource_variable_ops.is_resource_variable(obj)):
    return function_saved_model_utils.TrackableConstant(obj, parent)
  if not isinstance(obj, base.Trackable):
    raise ValueError(f"Cannot convert {obj} to Trackable.")
  return obj
