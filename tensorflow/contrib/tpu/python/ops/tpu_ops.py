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
# =============================================================================

"""Operations for TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform

from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging

if platform.system() != "Windows":
  # pylint: disable=wildcard-import,unused-import,g-import-not-at-top
  from tensorflow.contrib.tpu.ops import gen_tpu_ops
  from tensorflow.contrib.tpu.ops.gen_tpu_ops import *

  from tensorflow.contrib.util import loader
  from tensorflow.python.platform import resource_loader
  # pylint: enable=wildcard-import,unused-import,g-import-not-at-top

  _tpu_ops = loader.load_op_library(
      resource_loader.get_path_to_datafile("_tpu_ops.so"))

  def _create_default_group_assignment():
    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards is None:
      logging.warning(
          "cross_replica_sum should be used within a tpu_shard_context, but "
          "got unset number_of_shards. Assuming 1.")
      num_shards = 1
    group_assignment = [list(range(num_shards))]
    return group_assignment

  def all_to_all(x,
                 concat_dimension,
                 split_dimension,
                 split_count,
                 group_assignment=None,
                 name=None):
    """Exchange data across TPU replicas.

    Args:
      x: The local tensor.
      concat_dimension: The dimension number to concatenate.
      split_dimension: The dimension number to split.
      split_count: The number of splits, this number must equal to the sub-group
        size(group_assignment.get_shape()[1])
      group_assignment: Optional 2d int32 lists with shape [num_groups,
        num_replicas_per_group]. `group_assignment[i]` represents the replica
        ids in the ith subgroup.
      name: Optional op name.

    Returns:
      A `Tensor` which is concatenated by data from different replicas.
    """
    if group_assignment is None:
      group_assignment = _create_default_group_assignment()
    return gen_tpu_ops.all_to_all(
        x,
        group_assignment,
        concat_dimension=concat_dimension,
        split_dimension=split_dimension,
        split_count=split_count,
        name=name)

  @ops.RegisterGradient("AllToAll")
  def _all_to_all_grad(op, grad):
    # The gradient of a all-to-all is also a all-to-all but the
    # split_dimension and concat_dimension is swapped.
    # The graident with respect to group_assignment is None.
    return [
        gen_tpu_ops.all_to_all(
            grad,
            op.inputs[1],
            concat_dimension=op.get_attr("split_dimension"),
            split_dimension=op.get_attr("concat_dimension"),
            split_count=op.get_attr("split_count")), None
    ]

  def cross_replica_sum(x, group_assignment=None, name=None):
    """Sum the input tensor accorss replicas according to group_assignment.

    Args:
      x: The local tensor to the sum.
      group_assignment: Optional 2d int32 lists with shape [num_groups,
        num_replicas_per_group]. `group_assignment[i]` represents the replica
        ids in the ith subgroup.
      name: Optional op name.

    Returns:
      A `Tensor` which is summed across replicas.
    """
    if group_assignment is None:
      group_assignment = _create_default_group_assignment()

    return gen_tpu_ops.cross_replica_sum(x, group_assignment, name=name)

  @ops.RegisterGradient("CrossReplicaSum")
  def _cross_replica_sum_grad(op, grad):
    # The gradient of a cross replica sum is also a cross-replica sum.
    # The graident with respect to group_assignment is None.
    return [gen_tpu_ops.cross_replica_sum(grad, op.inputs[1]), None]

  # This extra type checking exists to give a more helpful error message in
  # the common case that uint8 and int64 values are infed. Remove when both
  # types are supported.

  _SUPPORTED_INFEED_DTYPES = set([
      dtypes.bool, dtypes.int32, dtypes.int64, dtypes.bfloat16, dtypes.float32,
      dtypes.complex64
  ])

  def infeed_dequeue(dtype, shape, name=None):
    """A placeholder op for a value that will be fed into the computation.

    Args:
      dtype: A `tf.DType`. The type of elements in the tensor.
      shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `dtype`.
      A tensor that will be provided using the infeed mechanism.

    Raises:
      TypeError: If 'dtype` is not a supported infeed type.
    """
    if dtype not in _SUPPORTED_INFEED_DTYPES:
      raise TypeError(
          "{} is not a supported TPU infeed type. Supported types are: "
          "{}".format(dtype, list(_SUPPORTED_INFEED_DTYPES)))

    return gen_tpu_ops.infeed_dequeue(dtype, shape, name=name)

  # pylint: disable=redefined-outer-name
  def infeed_dequeue_tuple(dtypes, shapes, name=None):
    """A placeholder op for values fed into the TPU simultaneously as a tuple.

    Args:
      dtypes: A list of `tf.DType`s that has length `>= 1`.
        The element types of each element in `outputs`.
      shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
        The shapes of each tensor in `outputs`.
      name: A name for the operation (optional).

    Returns:
      A list of `Tensor` objects of type `dtypes`.
      A list of tensors that will be provided using the infeed mechanism.

    Raises:
      TypeError: If a type in 'dtypes` is not a supported infeed type.
    """
    for dtype in dtypes:
      if dtype not in _SUPPORTED_INFEED_DTYPES:
        raise TypeError(
            "{} is not a supported TPU infeed type. Supported types are: "
            "{}".format(dtype, list(_SUPPORTED_INFEED_DTYPES)))
    return gen_tpu_ops.infeed_dequeue_tuple(dtypes, shapes, name=name)
  # pylint: enable=redefined-outer-name

else:
  # We have already built the appropriate libraries into the binary via CMake
  # if we have built contrib, so we don't need this
  pass
