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
"""ShardedVariable class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base as trackable


class ShardedVariable(trackable.Trackable):
  """A container for `Variables` that should be treated as shards.

  Variables that are too large to fit on a single device (e.g., large
  embeddings)
  may need to be sharded over multiple devices. This class maintains a list of
  smaller variables that can be independently stored on separate devices (eg,
  multiple parameter servers), and saves and restores those variables as if they
  were a single larger variable.

  Objects of this class can be saved with a given number of shards and then
  restored from a checkpoint into a different number of shards.

  Objects of this class can be saved to SavedModel format using
  `tf.saved_model.save`. The SavedModel can be used by programs like TF serving
  APIs. It is not yet supported to load the SavedModel with
  `tf.saved_model.load`.

  Since `ShardedVariable` can be saved and then restored to different number of
  shards depending on the restore environments, for example, TF serving APIs
  would restore to one shard for serving efficiency, when using
  `ShardedVariable` in a tf.function, one should generally not assume it has the
  same number of shards across save and load.

  Sharding is only supported along the first dimension.

  >>> class Model(tf.Module):
  ...   def __init__(self):
  ...     self.sharded_variable = ShardedVariable([
  ...       tf.Variable([3.0], dtype=tf.float32),
  ...       tf.Variable([2.0], dtype=tf.float32)
  ...     ])
  ...
  ...   @tf.function(input_signature=[tf.TensorSpec([], dtype=tf.int32)])
  ...   def fn(self, x):
  ...     return tf.nn.embedding_lookup(self.sharded_variable.variables, x)
  ...
  ...   @tf.function(input_signature=[tf.TensorSpec([], dtype=tf.int32)])
  ...   def serve_fn(self, x):
  ...     return tf.nn.embedding_lookup(self.sharded_variable.variables, x)
  >>>
  >>> model = Model()
  >>> model.fn(1).numpy()
  2.0
  >>> tf.saved_model.save(model, export_dir='/tmp/saved_model',
  ...   signatures=model.serve_fn)
  """

  def __init__(self, variables, name='ShardedVariable'):
    """Treats `variables` as shards of a larger Variable.


    Example:

    ```
    variables = [
      tf.Variable(..., shape=(10, 100), dtype=tf.float32),
      tf.Variable(..., shape=(15, 100), dtype=tf.float32),
      tf.Variable(..., shape=(5, 100), dtype=tf.float32)
    ]
    sharded_variable = ShardedVariable(variables)
    assert sharded_variable.shape.as_list() == [30, 100]
    ```

    Args:
      variables: A list of `ResourceVariable`s that comprise this sharded
        variable. Variables should not be shared between different
        `ShardedVariable` objects.
      name: String. Name of this container. Defaults to "ShardedVariable".
    """
    super(ShardedVariable, self).__init__()
    self._variables = variables
    self._name = name

    first_var = variables[0]

    if any(not isinstance(v, variables_lib.Variable) for v in variables):
      raise ValueError(
          'Expected a list of `Variable`s, found: {}'.format(variables))

    dtypes = {v.dtype for v in variables}
    if len(dtypes) > 1:
      raise ValueError(
          'All `Variable`s must have the same dtype, found: {}'.format(
              [v.dtype for v in variables]))
    self._dtype = first_var.dtype

    # All variables must have the same shape for axes > 0.
    higher_dim_shapes = {tuple(v.shape.as_list()[1:]) for v in variables}
    if len(higher_dim_shapes) > 1:
      raise ValueError(
          'All `Variables`s must have the same shapes except for the first '
          'axis, found {}'.format([v.shape for v in variables]))
    first_dim = sum(int(v.shape[0]) for v in variables)
    self._shape = tensor_shape.TensorShape([first_dim] + first_var.shape[1:])

    save_slice_info = [v._get_save_slice_info() for v in variables]  # pylint: disable=protected-access
    if any(slice_info is not None for slice_info in save_slice_info):
      raise ValueError('`SaveSliceInfo` should not be set for `Variable`s. '
                       '`ShardedVariable` will infer `SaveSliceInfo` according '
                       'to the order of the `Variable`s in the list passed to '
                       'the constructor. Found {}'.format(save_slice_info))

    # We create an uninitialized saving_variable with the full shape, which can
    # be later captured in signatures so that the signatures can treat this
    # ShardedVariable as one single variable.
    self._saving_variable = resource_variable_ops.UninitializedVariable(
        shape=self._shape, dtype=self._dtype, name=self._name)

  def __iter__(self):
    """Return an iterable for accessing the underlying sharded variables."""
    return iter(self._variables)

  @property
  def variables(self):
    """The list of `Variable`s that make up the shards of this object."""
    if save_context.in_save_context():
      return [self._saving_variable]
    return self._variables

  @property
  def name(self):
    """The name of this object. Used for checkpointing."""
    return self._name

  @property
  def dtype(self):
    """The dtype of all `Variable`s in this object."""
    return self._dtype

  @property
  def shape(self):
    """The overall shape, combining all shards along axis `0`."""
    return self._shape

  def _gather_saveables_for_checkpoint(self):
    """Return a `Saveable` for each shard. See `Trackable`."""

    def _saveable_factory(name=self.name):
      """Creates `SaveableObject`s for this `ShardedVariable`."""
      saveables = []
      dims = len(self._variables[0].shape)
      var_offset = [0 for _ in range(dims)]
      for v in self._variables:
        save_slice_info = variables_lib.Variable.SaveSliceInfo(
            full_name=self.name,
            full_shape=self.shape.as_list(),
            var_offset=copy.copy(var_offset),
            var_shape=v.shape.as_list())
        saveables.append(
            saveable_object_util.ResourceVariableSaveable(
                v, save_slice_info.spec, name))
        var_offset[0] += int(v.shape[0])
      return saveables

    return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

  def _map_resources(self):
    """For implementing `Trackable`."""
    obj_map, resource_map = {}, {}
    for v in self._variables + [self._saving_variable]:
      v_obj_map, v_resource_map = v._map_resources()  # pylint:disable=protected-access
      obj_map.update(v_obj_map)
      resource_map.update(v_resource_map)
    obj_map[self] = ShardedVariable([obj_map[self._saving_variable]],
                                    name=self.name)

    return obj_map, resource_map
