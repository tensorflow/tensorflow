# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for dealing with the analytical model inside of the iterator."""

from tensorflow.core.framework import model_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.get_model_proto")
def get_model_proto(iterator) -> model_pb2.ModelProto:
  """Gets the analytical model inside of `iterator` as `model_pb2.ModelProto`.

  Args:
    iterator: An `iterator_ops.OwnedIterator` or `dataset_ops.NumpyIterator`

  Returns:
    The model inside of this iterator as a model proto.

  Raises:
    NotFoundError: If this iterator's autotune is not enabled.
  """

  if isinstance(iterator, iterator_ops.OwnedIterator):
    iterator_resource = iterator._iterator_resource  # pylint: disable=protected-access
  elif isinstance(iterator, dataset_ops.NumpyIterator):
    iterator_resource = iterator._iterator._iterator_resource  # pylint: disable=protected-access
  else:
    raise ValueError("Only supports `tf.data.Iterator`-typed `iterator`.")

  if not context.executing_eagerly():
    raise ValueError(
        f"{get_model_proto.__name__} is not supported in graph mode."
    )

  model_proto_string_tensor = ged_ops.iterator_get_model_proto(
      iterator_resource
  )
  model_proto_bytes = model_proto_string_tensor.numpy()

  return model_pb2.ModelProto.FromString(model_proto_bytes)
