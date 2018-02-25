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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

if platform.system() != "Windows":
  # pylint: disable=wildcard-import,unused-import,g-import-not-at-top
  from tensorflow.contrib.tpu.ops import gen_tpu_ops
  from tensorflow.contrib.tpu.ops.gen_tpu_ops import *

  from tensorflow.contrib.util import loader
  from tensorflow.python.platform import resource_loader
  # pylint: enable=wildcard-import,unused-import,g-import-not-at-top

  _tpu_ops = loader.load_op_library(
      resource_loader.get_path_to_datafile("_tpu_ops.so"))

  @ops.RegisterGradient("CrossReplicaSum")
  def _cross_replica_sum_grad(op, grad):
    del op  # Unused
    # The gradient of a cross replica sum is also a cross-replica sum.
    return gen_tpu_ops.cross_replica_sum(grad)

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
