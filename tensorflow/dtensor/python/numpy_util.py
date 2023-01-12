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
"""Utilities to convert data buffers to/from DTensor tensors."""
from typing import List

import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.types.core import Tensor, TensorLike  # pylint: disable=g-multiple-import

# FIXME(b/262894693): Functions in this file are buggy.
# They do not distinguish between the client-local data and the global view.


def _split(value, splits, axis=0, split_fn=np.split, stack_fn=np.stack):
  """Split `value` into a sharded nparray/tf tensor based on the number of splits.
  """
  children = split_fn(value, splits[0], axis=axis)
  if len(splits) > 1:
    splits = splits[1:]
    children = [_split(child, splits, axis + 1) for child in children]
  return stack_fn(children)


def to_numpy(tensor: TensorLike) -> np.ndarray:
  """Copy `input` DTensor to an equivalent local numpy array."""
  layout = api.fetch_layout(tensor)
  if layout.mesh.is_remote():
    return np.array([None])

  unpacked = [tensor.numpy() for tensor in api.unpack(tensor)]
  return unpacked_to_numpy(unpacked, layout)


def unpacked_to_numpy(unpacked: List[TensorLike],
                      layout: layout_lib.Layout) -> np.ndarray:
  """Heals local Tensor components to a numpy array."""
  if len(unpacked) != len(layout.offset_to_shard()):
    raise ValueError('Wrong number of component Tensors.')

  unravelled = np.ndarray([layout.num_shards(i) for i in range(layout.rank)],
                          dtype=object)

  for offset, loc in enumerate(layout.offset_to_shard()):
    unravelled[loc] = unpacked[offset]

  concat_tensor = np.block(unravelled.tolist())

  # np.block can introduce empty initial dimensions, peel these off until
  # the output matches the rank of the input tensors.
  while concat_tensor.ndim > unpacked[0].ndim:
    concat_tensor = np.squeeze(concat_tensor, axis=0)
  return concat_tensor


# TODO(feyu): rename to slice.
def unpack(t: TensorLike,
           layout: layout_lib.Layout,
           split_fn=np.split,
           stack_fn=np.stack) -> List[TensorLike]:
  """Slice `t` into a flattened list of tensors suitable for `pack`."""
  if not layout.rank:
    return [t] * layout.mesh.size
  sharded_tensor = _split(
      t, [layout.num_shards(i) for i in range(layout.rank)],
      split_fn=split_fn,
      stack_fn=stack_fn)
  flattened = [np.ndarray([])] * layout.mesh.size
  for offset, shard in enumerate(layout.offset_to_shard()):
    flattened[offset] = sharded_tensor[tuple(shard)]
  return flattened


def pack_numpy(value: np.ndarray,
               layout: layout_lib.Layout,
               make_sparse: bool = False) -> Tensor:
  assert value is not None
  unpacked = unpack(value, layout)
  if make_sparse:
    return api.pack([sparse_ops.from_dense(t) for t in unpacked], layout)
  return api.pack(unpacked, layout)


def pack_tf_tensor(value: Tensor, layout: layout_lib.Layout) -> Tensor:
  if value is None:
    raise ValueError('pack requires values to be passed in')
  unpacked = unpack(
      value, layout, split_fn=array_ops.split, stack_fn=array_ops.stack)
  return api.pack(unpacked, layout)
