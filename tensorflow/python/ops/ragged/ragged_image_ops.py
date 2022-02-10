# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Image operations for RaggedTensors."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch


@dispatch.dispatch_for_api(image_ops.resize_images_v2)
def resize_images_v2(images: ragged_tensor.RaggedTensor,
                     size,
                     method=image_ops.ResizeMethod.BILINEAR,
                     preserve_aspect_ratio=False,
                     antialias=False,
                     name=None):
  """RaggedTensor dispatcher for tf.image.resize (tf-v2)."""
  with ops.name_scope(name, "RaggedResizeImages", [images, size]):
    return _resize_images(
        image_ops.resize_images_v2,
        images,
        size,
        method=method,
        preserve_aspect_ratio=preserve_aspect_ratio,
        antialias=antialias)


@dispatch.dispatch_for_api(image_ops.resize_images)
def resize_images_v1(images: ragged_tensor.RaggedTensor,
                     size,
                     method=image_ops.ResizeMethodV1.BILINEAR,
                     align_corners=False,
                     preserve_aspect_ratio=False,
                     name=None):
  """RaggedTensor dispatcher for tf.image.resize (tf-v1)."""
  with ops.name_scope(name, "RaggedResizeImages", [images, size]):
    return _resize_images(
        image_ops.resize_images,
        images,
        size,
        method=method,
        preserve_aspect_ratio=preserve_aspect_ratio,
        align_corners=align_corners)


def _resize_images(resize_op, images, size, **kwargs):
  """RaggedTensor dispatcher for tf.image.resize."""
  if images.shape.rank != 4:
    raise ValueError(
        "tf.image.resize: images.shape.rank must be 4 if images is ragged.")

  # Determine the output shape (excluding the batch dimension).
  static_batch_size = tensor_shape.dimension_value(images.shape[0])
  size = ops.convert_to_tensor(size, dtypes.int32, "size")
  size_as_shape = tensor_util.constant_value_as_shape(size).with_rank(2)
  out_shape = size_as_shape + images.shape[-1:]
  out_spec = tensor_spec.TensorSpec(out_shape, dtypes.float32)

  def resize_one(image):
    if isinstance(image, ragged_tensor.RaggedTensor):
      image = image.to_tensor()
    return resize_op(image, size, **kwargs)

  def resize_with_map():
    return map_fn.map_fn_v2(resize_one, images, fn_output_signature=out_spec)

  def empty_result():
    channels = array_ops.shape(images.flat_values)[-1:]
    return array_ops.zeros(array_ops.concat([[0], size, channels], axis=0))

  if static_batch_size == 0:
    return empty_result()
  elif static_batch_size is not None:
    return resize_with_map()
  else:
    empty_batch = math_ops.equal(images.nrows(), 0)
    return control_flow_ops.cond(empty_batch, empty_result, resize_with_map)
