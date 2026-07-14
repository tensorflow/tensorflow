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
"""`Callable` class used for checkpointing."""

from tensorflow.python.training.saving import saveable_object


class Callable(saveable_object.SaveSpec):
  """A callable that represents a Tensor that should be saved to checkpoint.

  This can be returned from `_serialize_to_tensor` in place of a Tensor. The
  callable will be executed on the specified device when the checkpoint is
  about to be written.

  Any class can use `Callable` for checkpointing, but for SavedModel export,
  only resource-type variables* are supported.

  * `resource_variable_ops.is_resource_variable(obj)` must return True.
  """

  def __init__(self, tensor_callable, dtype, device):
    """Initializes a `Callable` object.

    Args:
      tensor_callable: A callable that takes no arguments and returns a Tensor.
      dtype: Dtype of the tensor returned by the callable.
      device: Device of the tensor returned by the callable.
    """
    super().__init__(tensor_callable, None, None, dtype, device)
