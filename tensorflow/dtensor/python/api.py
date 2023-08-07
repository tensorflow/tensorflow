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
"""Core DTensor Python API."""

import contextlib
import threading
from typing import Any, Callable, Optional, Sequence

from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

_dtensor_singleton = None
_dtensor_singleton_lock = threading.Lock()

# -----------------------------------------------------------------------------
# Main methods to launch DTensor computations.


@tf_export("experimental.dtensor.call_with_layout", v1=[])
def call_with_layout(fn: Callable[...,
                                  Any], layout: Optional[layout_lib.Layout],
                     *args, **kwargs) -> Any:
  """Calls a function in the DTensor device scope if `layout` is not None.

  If `layout` is not None, `fn` consumes DTensor(s) as input and produces a
  DTensor as output; a DTensor is a tf.Tensor with layout-related attributes.

  If `layout` is None, `fn` consumes and produces regular tf.Tensors.

  Args:
    fn: A supported TF API function such as tf.zeros.
    layout: Optional, the layout of the output DTensor.
    *args:  Arguments given to `fn`.
    **kwargs: Keyword arguments given to `fn`.

  Returns:
    The return value of `fn` transformed to a DTensor if requested.
  """
  if layout is not None:
    if context.executing_eagerly():
      with default_mesh(layout.mesh):
        with _dtensor_device()._default_layout(layout):  # pylint: disable=protected-access
          return fn(*args, **kwargs)
    else:
      return relayout(fn(*args, **kwargs), layout)
  return fn(*args, **kwargs)


@tf_export("experimental.dtensor.run_on", v1=[])
@deprecation.deprecated(None, "Use `dtensor.default_mesh` scope instead.")
@contextlib.contextmanager
def run_on(mesh: layout_lib.Mesh):
  """Runs enclosed functions in the DTensor device scope.

  This function returns a scope. All the ops and tf.functions in this scope will
  run on the DTensor device using the mesh provided.
  This is useful for wrapping any tf.function that doesn't take a DTensor as
  input but would like to produce DTensor as result. The scope will also make
  sure all small constants be replicated as DTensor.

  Args:
    mesh: A Mesh instance to extract a default mesh from.

  Yields:
    A context in which all ops and tf.functions will run on the DTensor device.
  """
  with default_mesh(mesh):
    yield


@tf_export("experimental.dtensor.default_mesh", v1=[])
@contextlib.contextmanager
def default_mesh(mesh: layout_lib.Mesh):
  """Sets the default DTensor device mesh to use for enclosed functions.

  This function returns a scope. All the ops and tf.functions in this scope will
  default to this DTensor mesh if a mesh cannot be inferred from any of the
  inputs
  This is useful for wrapping any tf.function that doesn't take a DTensor as
  input but would like to produce DTensor as result. The scope will also make
  sure all small constants are replicated as DTensors.

  Args:
    mesh: A Mesh instance to extract a default mesh from.

  Yields:
    A context in which all ops and tf.functions will run on the given mesh.
  """
  if not isinstance(mesh, layout_lib.Mesh):
    raise ValueError(f"Expect `mesh` to be `Mesh`, got {type(mesh)}")

  with _dtensor_device()._experimental_default_mesh(mesh):  # pylint: disable=protected-access
    with ops.device(device_name()):
      yield


@tf_export("experimental.dtensor.get_default_mesh", v1=[])
def get_default_mesh() -> Optional[layout_lib.Mesh]:
  """Return the default mesh under the current dtensor device context.

  In the case that dtensor device system is not initialized, this function
  will return None.

  Returns:
    The current default mesh for the dtensor device context.
  """
  if _dtensor_singleton is None:
    return None
  else:
    return _dtensor_singleton._current_default_mesh   # pylint: disable=protected-access


@tf_export("experimental.dtensor.device_name", v1=[])
def device_name() -> str:
  """Returns the singleton DTensor device's name.

  This function can be used in the following way:

  ```python
  import tensorflow as tf

  with tf.device(dtensor.device_name()):
    # ...
  ```
  """
  return _dtensor_device().name


@tf_export("experimental.dtensor.is_dtensor", v1=[])
def is_dtensor(tensor) -> bool:
  """Check whether the input tensor is a DTensor.

  In Python, a DTensor has the same type as a `tf.Tensor`. This method will
  let you check and handle the tensor differently if a tf.Tensor is a DTensor.

  Args:
    tensor: an object to be checked.

  Returns:
    bool, True if the given tensor is a DTensor.
  """
  return _dtensor_device().is_dtensor(tensor)


# -----------------------------------------------------------------------------
# Data transfer methods.


@tf_export("experimental.dtensor.copy_to_mesh", v1=[])
def copy_to_mesh(
    tensor: Any,
    layout: layout_lib.Layout,
    source_layout: Optional[layout_lib.Layout] = None) -> tensor_lib.Tensor:
  """Copies a tf.Tensor onto the DTensor device with the given layout.

  Copies a regular tf.Tensor onto the DTensor device. Use the mesh attached to
  `layout` as target mesh. This method currently only supports replicated
  layouts, or one-to-one copies for sharded layouts.

  Args:
    tensor: A regular tf.Tensor to be copied as a DTensor.
    layout: Target layout (and mesh) for the result DTensor.
    source_layout: Source layout of the tensor before copy. This argument
      is deprecated.

  Returns:
    A DTensor on the DTensor device with the given layout.
  """
  del source_layout
  return relayout(tensor, layout)


@tf_export("experimental.dtensor.pack", v1=[])
def pack(tensors: Sequence[Any], layout: layout_lib.Layout) -> Any:
  """Packs `tf.Tensor` components into a DTensor.

  Packing and unpacking are inverse operations:

  ```
  * unpack(pack(tensors)) == tensors
  * pack(unpack(dtensor)) == dtensor
  ```

  1. For any DTensor on the mesh, `unpack` returns the raw components placed on
     each underlying device.
  2. Packing these raw components in the same order using `pack` returns a
     DTensor which should be identical to the original DTensor--both the content
     value and the layout.

  **Shape, Rank, and Scalars**: The rank of the DTensor is the same as the
  rank of its raw components, i.e., rank is preserved.  This leads to a
  consistent interpretation for packing scalar values into a DTensor. The only
  valid layout for a scalar value is fully replicated, and the individual
  components must be identical scalars.

  Each input `tensors[i]` will be copied to `layout.mesh.local_device[i]`
  if not already on the local device. Non-local components should not be passed
  to `pack`; use `copy_to_mesh` and `relayout` to place tensors on all global
  devices on a mesh.

  It is the caller's responsibility to ensure that the underlying values
  for `pack` adhere to the specified layout, and that only as many values are
  specified as there are local devices. Pack does not move data between clients.
  See examples below for more detail about layouts.

  For example, assume we have a mesh `[X(2), Y(3)]`, which has in total 6
  underlying devices. Futuremore, assume that the device location mapping is
  the following:

  ```
  device_ID  |  location X, Y
          0     0, 0
          1     0, 1
          2     0, 2
          3     1, 0
          4     1, 1
          5     1, 2
  ```

  1. For 1-D vector DTensor with shape `[128]` with layout `[mesh.X]` and value
     as `range(128)`, the raw components will have shape `[64]` each, and the
     raw components will be:

     ```
     device_ID  |  raw component
             0     range(0, 64)
             1     range(0, 64)
             2     range(0, 64)
             3     range(64, 128)
             4     range(64, 128)
             5     range(64, 128)
     ```

     This also means for a 1-D DTensor with shape `[2]` and layout `[mesh.X]`,
     the raw components have shape `[1]` rather than the shape for scalar values
     `[]`.

  2. For 2-D vector DTensor with shape `[2, 3]` with layout `[mesh.X, mesh.Y]`
     and value as `range(6)`, this is basically a fully-sharded DTensor.

     From global view, the content looks like
     ```
     [
       [0.0, 1.0, 2.0],
       [3.0, 4.0, 5.0],
     ]
     ```

     The raw components will have shape `[1, 1]` each, and have the following
     content:

     ```
     device_ID  |  raw component
             0     [[0.0]]
             1     [[1.0]]
             2     [[2.0]]
             3     [[3.0]]
             4     [[4.0]]
             5     [[5.0]]
     ```

  3. For a scalar value `123.0` DTensor, it can only have one legitimate layout
     `[]` (no dimension, but fully replicated).

     The raw components will have shape `[]` each, and have the following
     content:

     ```
     device_ID  |  raw component
             0     123.0
             1     123.0
             2     123.0
             3     123.0
             4     123.0
             5     123.0
     ```

     Again, caller of `pack` is expected to provide 6 identical value raw
     components with scalar shapes.

  4. For 3-D vector DTensor with shape `[2, 2, 3]` with layout
     `[X, unsharded, unsharded]` and value as `range(12)`,

     From global view, the content looks like:
     ```
     [
       [
         [0.0, 1.0, 2.0],
         [3.0, 4.0, 5.0],
       ],
       [
         [6.0, 7.0, 8.0],
         [9.0, 10., 11.],
       ],
     ]
     ```

     The raw components will have shape `[1, 2, 3]` each, and have the following
     content:

     ```
     device_ID  |  raw component
             0     range(6).reshape([1, 2, 3])
             1     range(6).reshape([1, 2, 3])
             2     range(6).reshape([1, 2, 3])
             3     range(6, 12).reshape([1, 2, 3])
             4     range(6, 12).reshape([1, 2, 3])
             5     range(6, 12).reshape([1, 2, 3])
     ```

  Args:
    tensors: The list of local tensor components to pack into a DTensor.
    layout: The layout of the DTensor to be created.

  Returns:
    A DTensor created from the individual component tensors.

  Raises:
    RuntimeError: When `pack` is not called eagerly.
  """
  return _dtensor_device().pack(tensors, layout)


@tf_export("experimental.dtensor.unpack", v1=[])
def unpack(tensor: Any) -> Sequence[Any]:
  """Unpacks a DTensor into `tf.Tensor` components.

  Packing and unpacking are inverse operations:

  ```
  * unpack(pack(tensors)) == tensors
  * pack(unpack(dtensor)) == dtensor
  ```

  1. For any DTensor on the mesh, `unpack` returns the raw components placed on
     each underlying device.
  2. Packing these raw components in the same order using `pack` returns a
     DTensor which should be identical to the original DTensor--both the content
     value and the layout.

  See the documentation for `pack` for more information about how packing and
  unpacking works.

  Args:
    tensor: The DTensor to unpack.

  Returns:
    The individual component tensors of the DTensor. This will include only the
    client-local components, i.e. the components placed on the local devices.

  Raises:
    RuntimeError: When `unpack` is not called eagerly.
  """
  return _dtensor_device().unpack(tensor)


# -----------------------------------------------------------------------------
# Layout-related methods.


@tf_export("experimental.dtensor.fetch_layout", v1=[])
def fetch_layout(tensor: tensor_lib.Tensor) -> layout_lib.Layout:
  """Fetches the layout of a DTensor.

  Args:
    tensor: The DTensor whose layout is to be fetched.

  Returns:
    The `Layout` of this DTensor.

  Raises:
    RuntimeError: When not called eagerly.
  """
  return _dtensor_device().fetch_layout(tensor)


@tf_export("experimental.dtensor.check_layout", v1=[])
def check_layout(tensor: tensor_lib.Tensor, layout: layout_lib.Layout) -> None:
  """Asserts that the layout of the DTensor is `layout`.

  Args:
    tensor: A DTensor whose layout is to be checked.
    layout: The `Layout` to compare against.

  Raises:
    ValueError: If the layout of `tensor` does not match the supplied `layout`.
  """
  if fetch_layout(tensor) != layout:
    raise ValueError("Layout of tensor: " + str(fetch_layout(tensor)) +
                     ", did not match expected layout: " + str(layout))


@tf_export("experimental.dtensor.relayout", v1=[])
def relayout(
    tensor: tensor_lib.Tensor,
    layout: layout_lib.Layout,
    name: Optional[str] = None,
) -> tensor_lib.Tensor:
  """Changes the layout of `tensor`.

  Changes the layout of `tensor` to `layout`. This is used to fine-tune the
  behavior of ops following/connected to `tensor`, such as choosing one SPMD
  expansion pattern over another. This works by forward propagating `layout`
  to connected TensorFlow computation graphs during layout propagation.

  Currently, only converting layouts from replicated to sharded or sharded to
  replicated per mesh dimension is supported. That is, "x, y" -> "unsharded, y"
  is supported, while "x, y" -> "z, y" is not supported.

  We also support a special "match" sharding spec, which instructs the relayout
  to act as an identity operation with respect to any sharding on these
  mesh dimensions.

  Relayout is internally lowered to a set of Split and/or AllToAll ops. When
  tensor layouts are converted from replicated to sharded, the cost is
  comparatively low because we only insert Split ops and no cross-device
  communication is needed. However, when tensor layouts are converted from
  sharded to replicated, cross-device communication may occur, causing potential
  performance impact.

  Args:
    tensor: A DTensor to specify a new layout for.
    layout: A Layout object specifying a new sharding spec.
    name: name of the Op.

  Returns:
    A DTensor output from the Relayout op.
  """
  layout_str = layout.to_string()
  with default_mesh(layout.mesh):
    return gen_dtensor_ops.relayout(tensor, layout_str, name=name)


@tf_export("experimental.dtensor.relayout_like", v1=[])
def relayout_like(
    tensor: tensor_lib.Tensor,
    layout_tensor: tensor_lib.Tensor,
    name: Optional[str] = None,
) -> tensor_lib.Tensor:
  """Changes the layout of `tensor` to the same as `layout_tensor`.

  `relayout_like` is often used inside a `tf.function`, to ensure a tensor is
  placed to the same mesh and with the same layout as another tensor.

  The backward gradient of a `relayout` is a `relayout_like` operation, to
  ensure the backward tensor has the same layout as the forward input tensor:

  ```
  @ops.RegisterGradient("Relayout")
  def _relayout_gradient(op, grad):
    return relayout_like(grad, layout_input=op.inputs[0])
  ```

  Here is another illustrative example:

  ```
  @tf.function
  def func(x):
    z = tf.ones(x.shape)
    z = dtensor.relayout_like(z, x)
    return x + z

  with dtensor.default_mesh(cpu_mesh):
    x = tf.ones((4, 4))

  with dtensor.default_mesh(gpu_mesh):
    y = func(x)

  # y would be on the cpu mesh, following the mesh of x.
  ```

  Args:
    tensor: A DTensor to specify a new layout for.
    layout_tensor: A Tensor object whose layout will be used for the layout of
      result. The shape and type of layout_tensor are irrelevant.
    name: name of the Op.

  Returns:
    A DTensor output from the RelayoutLike op.
  """

  return gen_dtensor_ops.relayout_like(
      input=tensor, layout_input=layout_tensor, name=name
  )


def _set_dtensor_device(device: dtensor_device.DTensorDevice) -> None:
  global _dtensor_singleton
  _dtensor_singleton = device


def _dtensor_device() -> dtensor_device.DTensorDevice:
  with _dtensor_singleton_lock:
    if _dtensor_singleton is None:
      _set_dtensor_device(
          dtensor_device.DTensorDevice(meshes=[], is_async=True))
  return _dtensor_singleton


def _reset() -> None:
  global _dtensor_singleton
  if _dtensor_singleton is not None:
    _dtensor_singleton.clear_tpu_core_ids()
  with _dtensor_singleton_lock:
    _dtensor_singleton = None


# ----------------------------------------------------------------------------
# Gradients


@ops.RegisterGradient("Relayout")
def _relayout_gradient(op, grad):
  grad = gen_dtensor_ops.relayout_like(grad, layout_input=op.inputs[0])
  return grad


@ops.RegisterGradient("RelayoutLike")
def _relayout_grad_gradient(op, grad):
  # Gradient of RelayoutGrad is relayout to the original Relayout's output.
  grad = gen_dtensor_ops.relayout_like(grad, layout_input=op.inputs[0])
  # Return None for forward_input's partial gradient since it is not connected
  # to the target's gradient.
  return grad, None


@ops.RegisterGradient("CopyToMesh")
def _copy_to_mesh_gradient(op, grad):
  grad = gen_dtensor_ops.copy_to_mesh_grad(
      grad,
      forward_input=op.inputs[0],
  )
  return grad


@ops.RegisterGradient("CopyToMeshGrad")
def _copy_to_mesh_grad_gradient(op, grad):
  grad = gen_dtensor_ops.copy_to_mesh_grad(
      grad,
      forward_input=op.inputs[0],
  )
  return grad, None
