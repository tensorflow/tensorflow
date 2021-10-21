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
"""Asset-type Trackable object."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.training.tracking import base
from tensorflow.python.util.tf_export import tf_export


@tf_export("saved_model.Asset")
class Asset(base.Trackable):
  """Represents a file asset to hermetically include in a SavedModel.

  A SavedModel can include arbitrary files, called assets, that are needed
  for its use. For example a vocabulary file used initialize a lookup table.

  When a trackable object is exported via `tf.saved_model.save()`, all the
  `Asset`s reachable from it are copied into the SavedModel assets directory.
  Upon loading, the assets and the serialized functions that depend on them
  will refer to the correct filepaths inside the SavedModel directory.

  Example:

  ```
  filename = tf.saved_model.Asset("file.txt")

  @tf.function(input_signature=[])
  def func():
    return tf.io.read_file(filename)

  trackable_obj = tf.train.Checkpoint()
  trackable_obj.func = func
  trackable_obj.filename = filename
  tf.saved_model.save(trackable_obj, "/tmp/saved_model")

  # The created SavedModel is hermetic, it does not depend on
  # the original file and can be moved to another path.
  tf.io.gfile.remove("file.txt")
  tf.io.gfile.rename("/tmp/saved_model", "/tmp/new_location")

  reloaded_obj = tf.saved_model.load("/tmp/new_location")
  print(reloaded_obj.func())
  ```

  Attributes:
    asset_path: A 0-D `tf.string` tensor with path to the asset.
  """

  def __init__(self, path):
    """Record the full path to the asset."""
    # The init_scope prevents functions from capturing `path` in an
    # initialization graph, since it is transient and should not end up in a
    # serialized function body.
    with ops.init_scope(), ops.device("CPU"):
      self._path = ops.convert_to_tensor(
          path, dtype=dtypes.string, name="asset_path")

  @property
  def asset_path(self):
    """Fetch the current asset path."""
    return self._path


ops.register_tensor_conversion_function(
    Asset, lambda asset, **kw: ops.convert_to_tensor(asset.asset_path, **kw))
