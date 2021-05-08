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
"""DLPack modules for Tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.util.tf_export import tf_export


@tf_export("experimental.dlpack.to_dlpack", v1=[])
def to_dlpack(tf_tensor):
  """Returns the dlpack capsule representing the tensor.

  This operation ensures the underlying data memory is ready when returns.

    ```python
    a = tf.tensor([1, 10])
    dlcapsule = tf.experimental.dlpack.to_dlpack(a)
    # dlcapsule represents the dlpack data structure
    ```

  Args:
    tf_tensor: Tensorflow eager tensor, to be converted to dlpack capsule.

  Returns:
    A PyCapsule named as dltensor, which shares the underlying memory to other
     framework. This PyCapsule can be consumed only once.
  """
  return pywrap_tfe.TFE_ToDlpackCapsule(tf_tensor)


@tf_export("experimental.dlpack.from_dlpack", v1=[])
def from_dlpack(dlcapsule):
  """Returns the Tensorflow eager tensor.

  The returned tensor uses the memory shared by dlpack capsules from other
  framework.

    ```python
    a = tf.experimental.dlpack.from_dlpack(dlcapsule)
    # `a` uses the memory shared by dlpack
    ```

  Args:
    dlcapsule: A PyCapsule named as dltensor

  Returns:
    A Tensorflow eager tensor
  """
  context.context().ensure_initialized()
  return pywrap_tfe.TFE_FromDlpackCapsule(dlcapsule, context.context()._handle)  # pylint: disable=protected-access
