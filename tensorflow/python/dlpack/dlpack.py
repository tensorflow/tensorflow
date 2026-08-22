# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""DLPack interop for TensorFlow.

DLPack (https://github.com/dmlc/dlpack) is a common in-memory tensor format
that lets frameworks like TensorFlow, PyTorch, and JAX share tensor data
without copying it. This module wraps the C++ conversion routines exposed
via ``pywrap_tfe`` and is the whole public surface of
``tf.experimental.dlpack``.
"""
from typing import Any

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("experimental.dlpack.to_dlpack", v1=[])
def to_dlpack(tf_tensor: "ops.Tensor") -> Any:
  """Returns the dlpack capsule representing the tensor.

  This operation ensures the underlying data memory is ready when it
  returns, so the capsule can be safely handed to another framework
  immediately.

  Example:

  ```python
  a = tf.constant([1, 10])
  dlcapsule = tf.experimental.dlpack.to_dlpack(a)
  # dlcapsule represents the dlpack data structure
  ```

  Args:
    tf_tensor: A TensorFlow eager tensor to convert to a dlpack capsule.

  Returns:
    A ``PyCapsule`` named ``"dltensor"`` that shares the tensor's underlying
    memory with another framework. The capsule can only be consumed once;
    pass it straight to that framework's ``from_dlpack``-equivalent call.
  """
  return pywrap_tfe.TFE_ToDlpackCapsule(tf_tensor)


@tf_export("experimental.dlpack.from_dlpack", v1=[])
def from_dlpack(dlcapsule: Any) -> "ops.Tensor":
  """Returns a TensorFlow eager tensor backed by a dlpack capsule's memory.

  The returned tensor shares its underlying memory with the capsule's
  original owner rather than copying it, so mutating one may affect the
  other depending on how the source framework manages the buffer.

  Example:

  ```python
  a = tf.experimental.dlpack.from_dlpack(dlcapsule)
  # `a` uses the memory shared by dlpack
  ```

  Args:
    dlcapsule: A ``PyCapsule`` named ``"dltensor"``, typically produced by
      another framework's dlpack export function (or by ``to_dlpack`` above).

  Returns:
    A TensorFlow eager tensor.
  """
  context.context().ensure_initialized()
  return pywrap_tfe.TFE_FromDlpackCapsule(
      dlcapsule, context.context()._handle)  # pylint: disable=protected-access
