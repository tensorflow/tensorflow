# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================
"""Experimental support for defining XLA shardings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.python_api import xla_shape
from tensorflow.core.framework import attr_value_pb2


class Sharding(object):
  """A class to support adding sharding attributes to Ops.

  Use the factory constructors and then call apply_to_tensor:
    Sharding.replicate().apply_to_tensor(tensor)
  """

  def __init__(self, proto=None):
    """Do not use this constructor; use the factory functions below."""
    self._proto = proto

  @classmethod
  def replicate(cls):
    """Returns a replicated sharding attribute.

    This causes an op to be computed in its entirety independently on all
    cores in the XLA device.
    """
    return Sharding(
        proto=xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED))

  @classmethod
  def assign_device(cls, core):
    """Returns an AssignDevice sharding attribute.

    This causes an op to be computed in its entirety only on one core in
    the XLA device.
    Args:
      core: The core to assign this Op to.
    """
    return Sharding(
        proto=xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.MAXIMAL,
            tile_assignment_dimensions=[1],
            tile_assignment_devices=[core]))

  @classmethod
  def tile(cls, tile_shape, tile_assignment):
    """Returns a Tiled sharding attribute.

    This causes an op to be partially computed on multiple cores in the
    XLA device.

    Args:
      tile_shape: A xla_shape.Shape describing the tile shape that each core
        will compute.
        The tile shape does not need to be divisible by the tile assignment.
      tile_assignment: An np.ndarray describing the topology of the tiling and
        which device will compute which part of the topology.

    Raises:
      TypeError: tile_assignment was not of np.array type or tile_shape was
         not of xla_shape.Shape type.

    TODO(jmolloy): This concept is nefarious and is not
    something we really want to expose to users (especially as the
    contract for tile_assignment is very strict).
    """
    if not isinstance(tile_assignment, np.ndarray):
      raise TypeError('Tile assignment must be of type np.ndarray')
    if not isinstance(tile_shape, xla_shape.Shape):
      raise TypeError('Tile shape must be of type xla_shape.Shape')
    dims = list(tile_assignment.shape)
    flattened_devices = tile_assignment.reshape(-1, order='C')
    return Sharding(
        proto=xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.OTHER,
            tile_shape=tile_shape.message,
            tile_assignment_dimensions=dims,
            tile_assignment_devices=list(flattened_devices)))

  @classmethod
  def split(cls, tensor, split_dimension, num_devices):
    """Returns a Sharding that splits a tensor across a dimension.

    This creates a Tiled attribute, similar to tile(), but easier to use for the
    common case of tiling a tensor N ways in one dimension.

    Args:
      tensor: A tf.Tensor to split.
      split_dimension: The dimension number to split.
      num_devices: The number of cores to split `tensor` over.

    Raises:
      ValueError: The tensor to split was smaller in the split dimension than
        the number of devices to split over.
    """
    tensor.shape.assert_is_fully_defined()
    shape = tensor.shape.as_list()
    if shape[split_dimension] < num_devices:
      raise ValueError('Split dimension was smaller than the required number '
                       'of splits: shape=%r, dimension=%r, num_devices=%r',
                       shape, split_dimension, num_devices)

    tile_shape = shape
    tile_shape[split_dimension] = int(
        math.ceil(tile_shape[split_dimension] / num_devices))
    tile_shape_proto = xla_data_pb2.Shape(
        element_type=xla_data_pb2.F32, dimensions=tile_shape)

    tile_assignment_dims = [1] * len(shape)
    tile_assignment_dims[split_dimension] = num_devices

    return Sharding(
        proto=xla_data_pb2.OpSharding(
            type=xla_data_pb2.OpSharding.OTHER,
            tile_shape=tile_shape_proto,
            tile_assignment_dimensions=tile_assignment_dims,
            tile_assignment_devices=range(num_devices)))

  def apply_to_tensor(self, tensor):
    """Applies this Sharding attribute to `tensor`."""
    if len(tensor.op.outputs) > 1:
      proto = self._get_or_create_tuple_proto(tensor.op)
      # We can't mutate an element of old_proto.tuple_shardings, so create
      # a new proto.
      tuple_shardings = list(proto.tuple_shardings)
      tuple_shardings[tensor.value_index] = self._proto
      proto = xla_data_pb2.OpSharding(
          type=xla_data_pb2.OpSharding.TUPLE, tuple_shardings=tuple_shardings)
    else:
      proto = self._proto

    attr_value = attr_value_pb2.AttrValue(s=proto.SerializeToString())
    # TODO(jmolloy): This need to be seriously revisited before declaring this
    # API available for public use.
    # pylint: disable=protected-access
    tensor.op._set_attr('_XlaSharding', attr_value)

  @property
  def proto(self):
    """Return the sharding protobuf of type xla_data_pb2.OpSharding."""
    return self._proto

  def _get_or_create_tuple_proto(self, op):
    try:
      attr = op.get_attr('_XlaSharding')
      proto = xla_data_pb2.OpSharding()
      proto.ParseFromString(attr)
      return proto
    except ValueError:
      return self._create_tuple_proto(op)

  def _create_tuple_proto(self, op):
    shardings = [
        xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED)
        for _ in op.outputs
    ]
    return xla_data_pb2.OpSharding(
        type=xla_data_pb2.OpSharding.TUPLE, tuple_shardings=shardings)


# Helpers for the above factory functions that allow easy application of
# shardings, for example:
#   tensor = xla_sharding.replicate(tensor)


def replicate(tensor):
  Sharding.replicate().apply_to_tensor(tensor)
  return tensor


def assign_device(tensor, device):
  Sharding.assign_device(device).apply_to_tensor(tensor)
  return tensor


def tile(tensor, tile_shape, tile_assignment):
  Sharding.tile(tile_shape, tile_assignment).apply_to_tensor(tensor)
  return tensor


def split(tensor, split_dimension, num_devices):
  Sharding.split(tensor, split_dimension, num_devices).apply_to_tensor(tensor)
  return tensor
