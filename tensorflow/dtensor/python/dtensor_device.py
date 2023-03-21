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
"""Propagates information about tensor layouts across operations."""

import contextlib
import logging
import threading
from typing import Any, List, Sequence, Set

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables


# TODO(allenl): Allow something other than "CUSTOM" so we don't need device
# numbering hacks to avoid collisions between parallel devices and dtensor
# devices.
_next_device_number = 0
_next_device_number_lock = threading.Lock()


class DTensorDevice(object):
  """Wraps a custom device which attempts to propagate tensor layouts."""

  def __init__(self,
               meshes: List[layout_lib.Mesh],
               is_async=True,
               in_flight_nodes_limit=8):
    """Create a new DTensorDevice which executes ops on `underlying_device`.

    Args:
      meshes: A list of `Mesh` objects indicating groups of devices to execute
        on. These may also be registered lazily.
      is_async: Indicates whether DTensor operations on this client will return
        immediately (with "non-ready" handles) or block until executed. This is
        on by default and is exposed as an option for ease of debugging.
      in_flight_nodes_limit: Indicates the limit of in-flight nodes before
        enqueueing of async operations to DTensorDevice is blocked. This limit
        is per mesh. 0 for no limits from DTensor. Default is 8.
    """
    if any(not isinstance(mesh, layout_lib.Mesh) for mesh in meshes):
      raise TypeError(
          "Expected a flat list of Mesh objects, got {}".format(meshes))
    global _next_device_number
    ctx = context.context()
    with _next_device_number_lock:
      self.name = "{}/device:CUSTOM:{}".format(ctx.host_address_space(),
                                               _next_device_number)
      _next_device_number += 1
    device, device_info = _pywrap_dtensor_device.Allocate(self.name)
    context.register_custom_device(device, self.name, device_info)

    self._device_info = device_info
    self._current_output_layout = None
    self._current_default_mesh = None
    self._is_async = is_async
    self._in_flight_nodes_limit = in_flight_nodes_limit
    self._meshes = set()
    self._mesh_lock = threading.Lock()
    for mesh in meshes:
      self._register_mesh(mesh)

  def _create_host_array(self, shape, host_id):
    """Returns ID and device lists that can be used to create a host mesh."""
    num_global_devices = np.prod(shape)
    global_device_ids = np.arange(num_global_devices).reshape(shape)
    local_device_list = [
        tf_device.DeviceSpec(
            job=config.full_job_name(), device_type="CPU", device_index=0)
    ]
    num_local_devices = len(local_device_list)
    local_device_ids = [
        x + host_id * num_local_devices for x in range(num_local_devices)
    ]
    return global_device_ids, local_device_ids, local_device_list

  def _create_embedding_host_mesh(self, tpu_mesh: layout_lib.Mesh):
    """Returns Embedding host mesh for each client."""
    if tpu_mesh.device_type().upper() != "TPU":
      raise ValueError("Must pass input of a tpu mesh.")

    # Global device ids are global host ids, while local device ids contains
    # local host id.

    ts_local_device_ids = []
    ts_local_devices = []
    for local_device_str in tpu_mesh.local_devices():
      # We only need to keep TPU:0 for each client.
      if not local_device_str.endswith("TPU:0"):
        continue

      device_spec = tf_device.DeviceSpec.from_string(local_device_str)
      ts_local_device_ids.append(device_spec.task)
      ts_local_devices.append(device_spec.replace(device_type="CPU"))

    if not ts_local_device_ids or not ts_local_device_ids:
      logging.info(
          "Cannot create tpu system mesh as %s has no `TPU:0` local device "
          "found", tpu_mesh.to_string())
      return None

    ts_global_device_ids = np.arange(config.num_clients())
    # TODO(zhonglinhan): parse global device specs as input when not None.
    return layout_lib.Mesh(
        dim_names=[tpu_mesh.dim_names[0]],  # 1D mesh.
        global_device_ids=ts_global_device_ids,
        local_device_ids=ts_local_device_ids,
        local_devices=ts_local_devices)

  def _register_mesh(self, mesh: layout_lib.Mesh):
    """Idempotently register `mesh` with the dtensor device."""
    with self._mesh_lock:
      if mesh not in self._meshes:
        _pywrap_dtensor_device.AddMesh(self._device_info, mesh.to_string(),
                                       self._is_async, False,
                                       self._in_flight_nodes_limit)
        self._meshes.add(mesh)
        if mesh.device_type().upper() == "TPU":
          logging.info(
              "Registering virtual 1:1 mapped host mesh %s for mesh %s",
              mesh.host_mesh().to_string(), mesh.to_string())
          _pywrap_dtensor_device.AddMesh(self._device_info,
                                         mesh.host_mesh().to_string(),
                                         self._is_async, True,
                                         self._in_flight_nodes_limit)
          self._meshes.add(mesh.host_mesh())
          embedding_host_mesh = self._create_embedding_host_mesh(mesh)
          if embedding_host_mesh:
            logging.info(
                "Registering embedding host mesh %s on each client for mesh %s",
                embedding_host_mesh.to_string(), mesh.to_string())
            _pywrap_dtensor_device.AddMesh(self._device_info,
                                           embedding_host_mesh.to_string(),
                                           self._is_async, False,
                                           self._in_flight_nodes_limit)
            self._meshes.add(embedding_host_mesh)

  @property
  def meshes(self) -> Set[layout_lib.Mesh]:
    return self._meshes

  def copy_to_mesh(self, tensor, new_layout) -> ops.Tensor:
    """Copy `tensor` to `device` with the given layout."""
    self._register_mesh(new_layout.mesh)
    with ops.device(self.name):
      return gen_dtensor_ops.copy_to_mesh(tensor, layout=new_layout.to_string())

  def pack(self, tensors: Sequence[Any], layout: layout_lib.Layout) -> Any:
    """Packs tensors into a DTensor handle on this DTensor device.

    Packing and unpacking are inverse operations:

    ```
    * unpack(pack(tensors)) == tensors
    * pack(unpack(dtensor)) == dtensor
    ```

    Refer to `dtensor.pack` for more information.

    Args:
      tensors: The list of tensors to pack into a DTensor.
      layout: The layout of the DTensor to be created.

    Returns:
      A DTensor created from the individual component tensors.

    Raises:
      RuntimeError: When not called eagerly.
    """
    if not context.executing_eagerly():
      raise RuntimeError("`pack` must be called eagerly.")
    if any(
        issubclass(type(t), resource_variable_ops.BaseResourceVariable)
        for t in tensors):
      raise TypeError(
          "Received Variable input to Pack, Variable is not supported.")
    self._register_mesh(layout.mesh)
    with ops.device(self.name):
      if all(isinstance(t, sparse_tensor.SparseTensor) for t in tensors):
        if not all(t.shape == tensors[0].shape for t in tensors):
          raise TypeError("All input SparseTensors to Pack must be same shape.")
        is_sparse = True
        tensors = [t.indices for t in tensors] + [t.values for t in tensors] + [
            ops.convert_to_tensor(t.shape, dtype=dtypes.int64) for t in tensors
        ]
      elif any(isinstance(t, sparse_tensor.SparseTensor) for t in tensors):
        raise TypeError("Cannot Pack SparseTensors with Tensors.")
      else:
        is_sparse = False
      try:
        return _pywrap_dtensor_device.Pack(
            context.context()._handle,  # pylint: disable=protected-access
            tensors,
            layout.to_string(),
            self._device_info,
            is_sparse)
      except core._NotOkStatusException as e:  # pylint: disable=protected-access
        raise core._status_to_exception(e) from None  # pylint: disable=protected-access

  def unpack(self, dtensor: Any) -> Sequence[Any]:
    """Unpacks a DTensor handle on this DTensor device.

    Packing and unpacking are inverse operations:

    ```
    * unpack(pack(tensors)) == tensors
    * pack(unpack(dtensor)) == dtensor
    ```

    Refer to `dtensor.unpack` for more information.

    Args:
      dtensor: The DTensor to unpack.

    Returns:
      The raw underlying tensor components of the DTensor.

    Raises:
      RuntimeError: When not called eagerly.
    """
    if not context.executing_eagerly():
      raise RuntimeError("`unpack` must be called eagerly.")
    if issubclass(type(dtensor), resource_variable_ops.BaseResourceVariable):
      raise TypeError(
          "Received Variable input to unpack, Variable is not supported.")
    try:
      tensors = _pywrap_dtensor_device.Unpack(
          context.context()._handle,  # pylint: disable=protected-access
          dtensor,
          self._device_info)
    except core._NotOkStatusException as e:  # pylint: disable=protected-access
      raise core._status_to_exception(e) from None  # pylint: disable=protected-access

    is_sparse = _pywrap_dtensor_device.IsSparseDTensor(
        context.context()._handle,  # pylint: disable=protected-access.
        dtensor,
        self._device_info)
    if is_sparse:
      result = []
      for i in range(len(tensors) // 3):
        result.append(
            sparse_tensor.SparseTensor(tensors[i],
                                       tensors[i + len(tensors) // 3],
                                       tensors[i + 2 * len(tensors) // 3]))
      return result
    else:
      return tensors

  def fetch_layout(self, dtensor: Any) -> layout_lib.Layout:
    """Fetches the layout of the DTensor.

    Args:
      dtensor: The DTensor whose layout is to be fetched.

    Returns:
      The `Layout` of this DTensor.

    Raises:
      RuntimeError: When not called eagerly.
    """
    if not context.executing_eagerly():
      raise RuntimeError("`fetch_layout` must be called eagerly.")
    if issubclass(type(dtensor), resource_variable_ops.BaseResourceVariable):
      dtensor = dtensor.read_value()
    try:
      layout_string = _pywrap_dtensor_device.FetchLayout(
          context.context()._handle,  # pylint: disable=protected-access
          dtensor,
          self._device_info)
    except core._NotOkStatusException as e:  # pylint: disable=protected-access
      raise core._status_to_exception(e) from None  # pylint: disable=protected-access
    return layout_lib.Layout.from_string(layout_string)

  def is_dtensor(self, tensor: Any) -> bool:
    """Check whether the input tensor is a DTensor.

    In Python, a DTensor has the same type as a `tf.Tensor`. This method will
    let you check and handle the tensor differently if a tf.Tensor is a DTensor.

    Args:
      tensor: an object to be checked.

    Returns:
      bool, True if the given tensor is a DTensor.

    Raises:
      RuntimeError: When not called eagerly.
    """
    if not context.executing_eagerly():
      raise RuntimeError("`is_dtensor` must be called eagerly.")
    if not tensor_util.is_tensor(tensor):
      return False
    if isinstance(tensor, variables.Variable):
      # Get the resource handle for tf.Variable
      tensor = tensor._handle   # pylint: disable=protected-access
    return _pywrap_dtensor_device.IsDTensor(
        context.context()._handle,  # pylint: disable=protected-access
        tensor,
        self._device_info,
    )

  def set_tpu_core_ids(self, mesh_name, tpu_core_ids):
    """Sets the singleton global device ID-to-physical core ID map.

    Args:
      mesh_name: The name of a mesh. If empty, set the default mapping.
      tpu_core_ids: TPU core IDs sorted by TF task/device ordinal.
    """
    _pywrap_dtensor_device.SetTPUCoreIDs(self._device_info, mesh_name,
                                         tpu_core_ids)

  def clear_tpu_core_ids(self):
    _pywrap_dtensor_device.ClearTPUCoreIDs(self._device_info)

  def tpu_core_ids_to_locations(self, tpu_core_ids):
    """Translates TPU core IDs to TPU core locations.

    Args:
      tpu_core_ids: A list of TPU core IDs. Each one is an unsigned integer.

    Returns:
      A list of corresponding TPU core locations.
    """
    return _pywrap_dtensor_device.TPUCoreIDsToLocations(
        context.context()._handle,  # pylint: disable=protected-access
        self._device_info,
        tpu_core_ids)

  def tpu_core_locations_to_ids(self, tpu_core_locations):
    """Translates TPU core locations to TPU core IDs.

    Args:
      tpu_core_locations: A list of TPU core locations. Each one is a list of
        four unsigned integers, [x, y, z, core].

    Returns:
      A list of corresponding TPU core IDs.
    """
    return _pywrap_dtensor_device.TPUCoreLocationsToIDs(
        context.context()._handle,  # pylint: disable=protected-access
        self._device_info,
        tpu_core_locations)

  def _get_function_cache_stats(self):
    """Returns the number of cache hit and miss for function compilation.

    Returns:
      A dictionary.
        'miss': number of cache misses;
        'hit': number of cache hits; and
        'size': size of cache;
      miss count.
    """
    return _pywrap_dtensor_device.GetFunctionCacheStats(
        context.context()._handle,  # pylint: disable=protected-access,
        self._device_info,
    )

  def set_iterator_element_layouts(self, iterator_resource_dtensor,
                                   layouts: List[layout_lib.Layout]):
    """Sets the element layouts on an iterator resource tensor.

    Args:
      iterator_resource_dtensor: a DTensor created by packing the individiual
        iterator resource tensors.
      layouts: the flattened list of layouts to be applied to the elements
        emitted by the iterator resource DTensor.
    """
    _pywrap_dtensor_device.SetIteratorElementLayouts(
        context.context()._handle,  # pylint: disable=protected-access
        iterator_resource_dtensor,
        [layout.to_string() for layout in layouts],
        self._device_info)

  @contextlib.contextmanager
  def _experimental_default_mesh(self, mesh: layout_lib.Mesh):
    """Sets a default mesh for all ops in the scope.

    Note: This is an internal helper method, which is not user facing api.

    Useful for requesting a specific mesh for ops which would have no inferred
    layout, e.g. tf.zeros.

    Args:
      mesh: A Mesh to be used for ops without Mesh.

    Yields:
      Nothing.
    """
    previous_default = self._current_default_mesh
    self._register_mesh(mesh)
    _pywrap_dtensor_device.ExperimentalSetDefaultMesh(
        self._device_info,
        mesh.to_string().encode("utf-8"))
    self._current_default_mesh = mesh
    yield
    _pywrap_dtensor_device.ExperimentalClearDefaultMesh(self._device_info)
    if previous_default:
      _pywrap_dtensor_device.ExperimentalSetDefaultMesh(
          self._device_info,
          previous_default.to_string().encode("utf-8"))
    self._current_default_mesh = previous_default

  @contextlib.contextmanager
  def _default_layout(self, layout: layout_lib.Layout):
    """Sets a default output layout for all ops in the scope.

    Note: This is an internal helper method, which is not user facing api.

    Useful for requesting a specific layout for ops which would have no inferred
    layout, e.g. tf.zeros.

    Caveats:

    - Currently only affects the first output of an op. For Op with multiple
      outputs, this does not support yet.

    - All Ops in the scope will be attached with the same layout. This might not
      be valid as the rank is different. The current suggestion is: Try to wrap
      the raw op wheneven possible.

    Args:
      layout: A Layout for the outputs of all operations in this scope.

    Yields:
      Nothing.
    """
    previous_default = None
    previous_graph_size = None
    graph = None

    self._register_mesh(layout.mesh)
    try:
      previous_default = self._current_output_layout
      self._current_output_layout = layout.to_string().encode("utf-8")
      _pywrap_dtensor_device.ExperimentalSetDefaultLayout(
          self._device_info, self._current_output_layout)
      if context.executing_eagerly():
        with ops.device(self.name):
          yield
      else:
        # Custom devices currently don't affect graph building, so we need a
        # separate way to indicate layouts.
        #
        # TODO(allenl): Remove this case once the DTensor device is active
        # during tracing.
        graph = ops.get_default_graph()
        previous_graph_size = len(graph.get_operations())
        yield
    finally:
      if graph is not None:
        # Tag operations added under this scope
        for operation in graph.get_operations()[previous_graph_size:]:
          # Set layout directly on the Op itself.
          operation._set_attr(  # pylint: disable=protected-access
              "_layout",
              attr_value_pb2.AttrValue(
                  list=attr_value_pb2.AttrValue.ListValue(
                      s=[self._current_output_layout])))
          operation._set_attr(  # pylint: disable=protected-access
              "_mesh",
              attr_value_pb2.AttrValue(
                  s=layout.mesh.to_string().encode("utf-8")))

      self._current_output_layout = previous_default
      if self._current_output_layout is None:
        _pywrap_dtensor_device.ExperimentalClearDefaultLayout(self._device_info)
      else:
        _pywrap_dtensor_device.ExperimentalSetDefaultLayout(
            self._device_info, self._current_output_layout.decode("utf-8"))
