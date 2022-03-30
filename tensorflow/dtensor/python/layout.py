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
"""Python definitions for Mesh and Layout."""

import collections
import itertools
from typing import List, Dict, Optional

import numpy as np

from tensorflow.dtensor.proto import layout_pb2
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export

# UNSHARDED indicates a tensor dimension is not sharded over any mesh dimension.
UNSHARDED = 'unsharded'
MATCH = 'match'

tf_export('experimental.dtensor.UNSHARDED', v1=[]).export_constant(
    __name__, 'UNSHARDED')
tf_export('experimental.dtensor.MATCH', v1=[]).export_constant(
    __name__, 'MATCH')

MeshDimension = collections.namedtuple('MeshDimension', ['name', 'size'])


@tf_export('experimental.dtensor.Mesh', v1=[])
class Mesh(object):
  """Represents a Mesh configuration over a certain list of Mesh Dimensions."""

  _dim_dict: Dict[str, MeshDimension]

  def __init__(self,
               dim_names: List[str],
               global_device_ids: np.ndarray,
               local_device_ids: List[int],
               local_devices: List[tf_device.DeviceSpec],
               mesh_name: str = '',
               global_devices: Optional[List[tf_device.DeviceSpec]] = None):
    """Builds a Mesh.

    The dim_names and global_device_ids arguments describe the dimension names
    and shape for the mesh.

    For example,
      dim_names = ('x', 'y'),
      global_device_ids = [[0, 1],
                           [2, 3],
                           [4, 5]]
    defines a 2D mesh of shape 3x2. A reduction over the 'x' dimension will
    reduce across columns (0, 2, 4), and a reduction over the 'y' dimension
    reduces across rows.

    Args:
      dim_names: A list of strings indicating dimension names.
      global_device_ids: An ndarray of global device IDs is used to compose
        DeviceSpecs describing the mesh. The shape of this array determines the
        size of each mesh dimension. Values in this array should increment
        sequentially from 0. This argument is the same for every DTensor client.
      local_device_ids: A list of local device IDs equal to a subset of values
        in global_device_ids. They indicate the position of local devices in the
        global mesh. Different DTensor clients must contain distinct
        local_device_ids contents. All local_device_ids from all DTensor clients
        must cover every element in global_device_ids.
      local_devices: The list of devices hosted locally. The elements correspond
        1:1 to those of local_device_ids.
      mesh_name: The name of the mesh. Currently, this is rarely used, and is
        mostly used to indicate whether it is a CPU, GPU, or TPU-based mesh.
      global_devices (optional): The list of global devices. Set when multiple
        device meshes are in use.
    """
    # Check if input args are valid.
    if not isinstance(global_device_ids, np.ndarray):
      raise ValueError('Variable global_device_ids must be an ndarray.')
    if global_device_ids.size == 0:
      raise ValueError('Variable global_device_ids must be non-empty.')
    flat_global_device_ids = global_device_ids.flatten()
    # global_device_ids are expected to be consecutive numbers.
    # LINT.IfChange
    distance = flat_global_device_ids[0]
    if any(
        (gid - i != distance) for i, gid in enumerate(flat_global_device_ids)):
      raise ValueError('global_device_ids must sequentially increase: %s' %
                       global_device_ids)
    # LINT.ThenChange(//tensorflow/dtensor/cc/dtensor_device.cc)

    if len(dim_names) != global_device_ids.ndim:
      raise ValueError(
          'Number of mesh dimensions does not match number of dimension names.')

    if not isinstance(local_device_ids, list):
      raise ValueError('Variable local_device_ids must be a list of integers.')

    if not isinstance(local_devices, list):
      raise ValueError('Variable local_devices must be a list of DeviceSpecs.')

    if global_devices and not isinstance(global_devices, list):
      raise ValueError('Variable global_devices must be a list of DeviceSpecs.')

    if not local_devices and not global_devices:
      raise ValueError('Empty list of devices not allowed.')

    local_devices_set = set(local_devices)
    local_device_only_contains_host_cpu = (
        len(local_devices_set) == 1 and
        list(local_devices_set)[0].device_type == 'CPU')
    if not local_device_only_contains_host_cpu and len(local_devices) != len(
        local_devices_set):
      raise ValueError('Duplicate devices found in mesh specification %s.' %
                       [d for d in local_devices if local_devices.count(d) > 1])

    if len(local_device_ids) != len(local_devices):
      raise ValueError(
          'Variable local_device_ids does not have same size as local_devices.')

    if len(local_device_ids) > len(np.ravel(global_device_ids)):
      raise ValueError('Cannot have more local than gobal device IDs.')

    device_types = set([device.device_type for device in local_devices])
    if not device_types:
      device_types = set([device.device_type for device in global_devices])
    if None in device_types:
      raise ValueError('device_type is required')
    if len(device_types) > 1:
      raise ValueError('Devices containing multiple device_types : %s' %
                       device_types)

    # Set object's state.
    self._device_type = device_types.pop()
    self._dim_names = dim_names
    self._dim_dict = {
        dim_name: MeshDimension(dim_name, global_device_ids.shape[i])
        for i, dim_name in enumerate(dim_names)
    }
    self._global_device_ids = global_device_ids
    self._local_device_ids = local_device_ids
    self._local_devices = local_devices
    self._global_devices = global_devices
    self._name = mesh_name

  @property
  def dim_names(self) -> List[str]:
    return self._dim_names

  @property
  def name(self) -> str:
    return self._name

  def is_remote(self) -> bool:
    return not self._local_device_ids and self._global_device_ids.size > 0

  def host_mesh(self):
    """Returns the 1-1 mapped host mesh."""
    if self.device_type().upper() == 'CPU':
      return self

    v_cpus_counts = len(tf_config.list_logical_devices('CPU'))
    if v_cpus_counts < len(self._local_devices):
      raise ValueError('Must have at least {0} virtual CPUs for mesh : {1}, '
                       'but got : {2} virtual CPUs.'.format(
                           len(self._local_devices), self.to_string(),
                           v_cpus_counts))
    device_array = np.asarray([
        spec.replace(device_type='CPU')
        for spec in self._local_devices
    ]).reshape((len(self._local_devices), 1))
    global_devices = None
    if self._global_devices:
      global_devices = [
          spec.replace(device_type='CPU') for spec in self._global_devices
      ]
    h_mesh = Mesh(
        self._dim_names,
        self._global_device_ids,
        self.local_device_ids(),
        np.ravel(device_array).tolist(),
        global_devices=global_devices)
    return h_mesh

  def device_type(self) -> str:
    return self._device_type

  def contains_dim(self, dim_name: str) -> bool:
    return dim_name in self._dim_dict

  def __contains__(self, dim_name: str) -> bool:
    return self.contains_dim(dim_name)

  def dim_size(self, dim_name: str) -> int:
    """Returns the size of a dimension."""
    if dim_name not in self._dim_dict.keys():
      raise ValueError(('"{dim_name}" not a dimension name in current mesh. ' +
                        'Dimension names: {dim_names}.').format(
                            dim_name=dim_name,
                            dim_names=list(self._dim_dict.keys())))
    return self._dim_dict[dim_name].size

  def unravel_index(self):
    """Returns a dictionary from device ID to {dim_name: dim_index}.

    For example, for a 3x2 mesh, return this:
      { 0: {'x': 0, 'y', 0},
        1: {'x': 0, 'y', 1},
        2: {'x': 1, 'y', 0},
        3: {'x': 1, 'y', 1},
        4: {'x': 2, 'y', 0},
        5: {'x': 2, 'y', 1} }.
    """
    idx_ranges = [
        range(self.dim_size(dim_name)) for dim_name in self._dim_names
    ]
    mesh_pos = itertools.product(*idx_ranges)
    mapping = {}
    for device_id, device_pos in enumerate(mesh_pos):
      device_loc = {}
      for dim_name, dim_index in zip(self._dim_names, device_pos):
        device_loc[dim_name] = dim_index
      mapping[device_id] = device_loc
    return mapping

  def min_global_device_id(self) -> int:
    """Returns the minimum global device ID."""
    # global_device_ids sequentially increases.
    return self._global_device_ids.flatten()[0]

  def local_device_ids(self) -> List[int]:
    """Returns a list of local device IDs."""
    return self._local_device_ids

  def local_device_locations(self) -> List[Dict[str, int]]:
    """Returns a list of local device locations.

    A device location is a dictionary from dimension names to indices on those
    dimensions.
    """
    mapping = self.unravel_index()
    return [mapping[device_id] for device_id in self.local_device_ids()]

  def local_devices(self) -> List[str]:
    """Returns a list of local device specs represented as strings."""
    return [d.to_string() for d in self._local_devices]

  def num_local_devices(self) -> int:
    """Returns the number of local devices."""
    return len(self._local_devices)

  def to_string(self) -> str:
    """Returns string representation of Mesh."""

    # Get proto representation
    mesh_proto = self.as_proto()
    # Separate individual elements with ','.
    name = mesh_proto.name
    dim_str = ','.join(
        dim.name + '=' + str(dim.size) for dim in mesh_proto.mesh_dimensions)
    global_ids = ','.join(str(id) for id in mesh_proto.global_device_ids)
    local_ids = ','.join(str(id) for id in mesh_proto.local_device_ids)
    devices = ','.join(dev for dev in mesh_proto.local_devices)
    components = [name, dim_str, global_ids, local_ids, devices]
    if mesh_proto.global_devices:
      global_devices = ','.join(dev for dev in mesh_proto.global_devices)
      components.append(global_devices)
    # Separate mesh components with '|'.
    return '|'.join(components)

  def as_proto(self) -> layout_pb2.MeshProto:
    """Returns mesh protobuffer."""

    mesh_proto = layout_pb2.MeshProto()

    mesh_proto.name = self._name

    for i, mesh_dimension in enumerate(self._dim_names):
      dim = mesh_proto.mesh_dimensions.add()
      dim.name = mesh_dimension
      dim.size = self._global_device_ids.shape[i]

    for d in np.ravel(self._global_device_ids):
      mesh_proto.global_device_ids.append(d)

    for d in self._local_device_ids:
      mesh_proto.local_device_ids.append(d)

    for d in self._local_devices:
      mesh_proto.local_devices.append(d.to_string())

    if self._global_devices:
      for d in self._global_devices:
        mesh_proto.global_devices.append(d.to_string())

    return mesh_proto

  @staticmethod
  def from_string(mesh_str: str) -> 'Mesh':
    """Construct a mesh instance from input `proto`."""
    # Separate elements of mesh.
    mesh_parts = mesh_str.split('|')
    global_dev_str = None
    if len(mesh_parts) == 5:
      name, mesh_dim_strs, global_id_str, local_id_str, dev_str = mesh_parts
    elif len(mesh_parts) == 6:
      (name, mesh_dim_strs, global_id_str, local_id_str, dev_str,
       global_dev_str) = mesh_parts
    else:
      raise ValueError('Invalid mesh string : %s' % mesh_str)

    # Load mesh proto.
    mesh_proto = layout_pb2.MeshProto()
    mesh_proto.name = name

    for mesh_dim_str in mesh_dim_strs.split(','):
      name, size_str = mesh_dim_str.split('=')
      dim = mesh_proto.mesh_dimensions.add()
      dim.name = name
      dim.size = int(size_str)

    for global_id in global_id_str.split(','):
      mesh_proto.global_device_ids.append(int(global_id))

    if local_id_str:
      for local_id in local_id_str.split(','):
        mesh_proto.local_device_ids.append(int(local_id))

    if dev_str:
      for dev in dev_str.split(','):
        mesh_proto.local_devices.append(dev)

    if global_dev_str:
      for dev in global_dev_str.split(','):
        mesh_proto.global_devices.append(dev)

    return Mesh.from_proto(mesh_proto)

  @staticmethod
  def from_proto(proto: layout_pb2.MeshProto) -> 'Mesh':
    """Construct a mesh instance from input `proto`."""
    shape = [dim.size for dim in proto.mesh_dimensions]

    # Convert global_device ids list back into array form
    global_device_ids = [int(d) for d in proto.global_device_ids]
    global_device_ids = np.asarray(global_device_ids).reshape(shape)

    # Construct local_device_ids list
    local_device_ids = [int(d) for d in proto.local_device_ids]

    # Convert local devices list back to array form
    local_devices = [
        tf_device.DeviceSpec.from_string(d) for d in proto.local_devices
    ]

    # Convert global devices list back to array form
    global_devices = [
        tf_device.DeviceSpec.from_string(d) for d in proto.global_devices
    ]

    name = proto.name
    dims = [dim.name for dim in proto.mesh_dimensions]
    return Mesh(dims, global_device_ids, local_device_ids, local_devices, name,
                global_devices)

  def shape(self) -> List[int]:
    return [self.dim_size(dim) for dim in self._dim_names]

  @property
  def size(self) -> int:
    return len(np.ravel(self._global_device_ids))

  def __getitem__(self, dim_name: str) -> MeshDimension:
    if dim_name not in self._dim_dict:
      raise KeyError(
          f'Dimension {dim_name} not defined in mesh: {self._dim_dict.keys()}')
    return self._dim_dict[dim_name]

  # TODO(b/168730933): Define a nicer mesh ID.
  def __hash__(self):
    return hash(self.as_proto().SerializeToString(deterministic=True))

  def __eq__(self, other):
    if not isinstance(other, type(self)) and not isinstance(self, type(other)):
      raise ValueError('comparing with type : {0} but expecting : {1}'.format(
          type(other), type(self)))
    return self.as_proto().SerializeToString() == other.as_proto(
    ).SerializeToString()


# TODO(hthu): Consider making this class immutable.
@tf_export('experimental.dtensor.Layout', v1=[])
class Layout(object):
  """Represents the layout information for a Tensor."""

  def __init__(self, sharding_specs: List[str], mesh: Mesh):
    """Builds a Layout from a list of dimension names and a Mesh.

    Args:
      sharding_specs: List of sharding specifications, each corresponding to a
        tensor dimension. Each specification (dim_sharding) can either be a mesh
        dimension or the special value UNSHARDED.
      mesh: A mesh configuration for the Tensor.

    Returns:
      A valid Layout built with given layout & mesh.
    """
    # Validate mesh
    if not isinstance(mesh, Mesh):
      raise ValueError('mesh is not a valid Mesh object.')

    # Validate sharding spec
    for _, dim_sharding in enumerate(sharding_specs):
      # If special value no need to check for uniqueness, just skip.
      if dim_sharding == UNSHARDED or dim_sharding == MATCH:
        continue
      # Check dim_sharding is unique.
      if sharding_specs.count(dim_sharding) > 1:
        raise ValueError(
            ('Mesh dimension {mesh_dim} was repeated in sharding ' +
             'specification {sharding_specs}. Mesh dimensions must be unique ' +
             'in a layout.').format(
                 mesh_dim=dim_sharding, sharding_specs=sharding_specs))
      # Check dim_sharding is mesh dimension.
      if dim_sharding not in mesh:
        raise ValueError(
            ('{dim_sharding}: A dimension sharding must either be a ' +
             'valid mesh dimension or UNSHARDED.').format(
                 dim_sharding=dim_sharding))

    # Set object's state
    self.sharding_specs = sharding_specs
    self.rank = len(sharding_specs)
    self.mesh = mesh
    self.shape = [self.num_shards(i) for i in range(self.rank)]

  @staticmethod
  def from_string(layout_str: str) -> 'Layout':
    """Parses layout string."""
    layout_parts = layout_str.split(' ')
    if len(layout_parts) != 2:
      raise ValueError(
          'layout string must contain two parts: specs and mesh. But got {}.'
          .format(layout_str))

    sharding_specs_str = layout_parts[0].replace('sharding_specs:', '')
    mesh_str = layout_parts[1].replace('mesh:', '')

    sharding_specs = sharding_specs_str.split(',')[:-1]

    mesh = Mesh.from_string(mesh_str)
    layout = Layout(sharding_specs, mesh)
    return layout

  @staticmethod
  def from_str(layout_str: bytes) -> 'Layout':
    layout_proto = layout_pb2.LayoutProto()
    layout_proto.ParseFromString(layout_str)
    sharding_specs = [
        sharding_spec.sharding_spec
        for sharding_spec in layout_proto.sharding_specs
    ]
    mesh = Mesh.from_proto(layout_proto.mesh_config)
    return Layout(sharding_specs, mesh)

  def offset_to_shard(self):
    """Mapping from offset in a flattened list to shard index."""
    unravel_index = self.mesh.unravel_index()
    locations = [None] * self.mesh.size
    for offset, mesh_loc in unravel_index.items():
      loc = []
      for dim_sharding in self.sharding_specs:
        if dim_sharding == UNSHARDED:
          loc.append(0)
        else:
          loc.append(mesh_loc[dim_sharding])
      locations[offset] = tuple(loc)
    return locations

  def offset_tuple_to_global_index(self, offset_tuple):
    """Mapping from offset to index in global tensor."""
    index = 0
    for i, o in enumerate(offset_tuple):
      m = 1
      for x in range(i + 1, self.rank):
        m = m * self.num_shards(x)
      index = index + m * o
    return index

  def unravel(self, unpacked_tensors: List[np.ndarray]) -> np.ndarray:
    """Convert a flattened list of shards into a sharded array."""
    unravelled = np.ndarray([self.num_shards(i) for i in range(self.rank)],
                            dtype=np.object)
    for offset, loc in enumerate(self.offset_to_shard()):
      unravelled[loc] = unpacked_tensors[offset]
    return unravelled

  def num_shards(self, idx: int) -> int:
    """Returns the number of shards for tensor dimension `idx`."""
    dim_sharding = self.sharding_specs[idx]
    if dim_sharding == UNSHARDED:
      return 1
    if dim_sharding == MATCH:
      return -1
    return self.mesh.dim_size(dim_sharding)

  def as_proto(self) -> layout_pb2.LayoutProto:
    """Create a proto representation of a layout."""
    layout_proto = layout_pb2.LayoutProto()

    for dim_sharding in self.sharding_specs:
      tensor_dim = layout_proto.sharding_specs.add()
      tensor_dim.sharding_spec = dim_sharding

    layout_proto.mesh_config.CopyFrom(self.mesh_proto())

    return layout_proto

  def mesh_proto(self) -> layout_pb2.MeshProto:
    return self.mesh.as_proto()

  def is_fully_replicated(self) -> bool:
    return all([self.num_shards(i) == 1 for i in range(self.rank)])

  # A layout with no sharding specs is acceptable, therefore we only check the
  # mesh.
  def to_string(self) -> str:
    """Returns string representation of Layout."""
    sharding_spec_str = 'sharding_specs:'
    # Add comma after each instruction.
    for spec in self.sharding_specs:
      sharding_spec_str += spec + ','

    mesh_str = 'mesh:' + self.mesh.to_string()
    return sharding_spec_str + ' ' + mesh_str

  def serialized_string(self) -> bytes:
    return self.as_proto().SerializeToString()

  def __eq__(self, other) -> bool:
    return self.serialized_string() == other.serialized_string()

  def __repr__(self) -> str:
    return str(self.as_proto())

  @staticmethod
  def replicated(mesh: Mesh, rank: int) -> 'Layout':
    """Returns a replicated layout of rank `rank`."""
    return Layout([UNSHARDED] * rank, mesh)

  @staticmethod
  def batch_sharded(mesh: Mesh, batch_dim: str, rank: int) -> 'Layout':
    """Returns a layout sharded on batch dimension."""
    return Layout([batch_dim] + [UNSHARDED] * (rank - 1), mesh)

  @staticmethod
  def inner_sharded(mesh: Mesh, inner_dim: str, rank: int) -> 'Layout':
    """Returns a layout sharded on inner dimension."""
    return Layout([UNSHARDED] * (rank - 1) + [inner_dim], mesh)

  def delete(self, dims: List[int]) -> 'Layout':
    """Returns the layout with the give dimensions deleted."""
    if not isinstance(dims, list):
      dims = [dims]
    new_specs = [
        spec for i, spec in enumerate(self.sharding_specs) if i not in dims]
    return Layout(new_specs, self.mesh)
