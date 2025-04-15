# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for multiple meshes in DTensor."""

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

Layout = layout_lib.Layout
Mesh = layout_lib.Mesh
UNSHARDED = layout_lib.UNSHARDED

# Makes a 1D mesh with dimension X(2).
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_DEVICES_IDS = test_util.create_device_ids_array((2,))
_ONE_D_CPU_MESH = Mesh(
    [_MESH_DIM_X],
    _DEVICES_IDS,
    np.ravel(_DEVICES_IDS).tolist(),
    test_util.create_device_list((2,), 'CPU'),
)
_ONE_D_GPU_MESH = Mesh(
    [_MESH_DIM_X],
    _DEVICES_IDS,
    np.ravel(_DEVICES_IDS).tolist(),
    test_util.create_device_list((2,), 'GPU'),
)
_ONE_D_TPU_MESH = Mesh(
    [_MESH_DIM_X],
    _DEVICES_IDS,
    np.ravel(_DEVICES_IDS).tolist(),
    test_util.create_device_list((2,), 'TPU'),
)
_ONE_D_CPU_MESH_Y = Mesh(
    [_MESH_DIM_Y],
    _DEVICES_IDS,
    np.ravel(_DEVICES_IDS).tolist(),
    test_util.create_device_list((2,), 'CPU'),
)

_OTHER_CPU_MESH = Mesh(
    [_MESH_DIM_X],
    _DEVICES_IDS[:1],
    np.ravel(_DEVICES_IDS[:1]).tolist(),
    test_util.create_device_list((1,), 'CPU'),
)


class MultiMeshTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(MultiMeshTest, self).setUp()
    self.first_mesh = _ONE_D_CPU_MESH
    if test_util.is_tpu_present():
      self.second_mesh = _ONE_D_TPU_MESH
    elif test_util.is_gpu_present():
      self.second_mesh = _ONE_D_GPU_MESH
    else:
      self.second_mesh = _ONE_D_CPU_MESH_Y

    device_type = config.preferred_device_type()
    if device_type != 'TPU':
      test_util.reset_logical_devices(device_type, 2)
    accelerator_util.initialize_accelerator_system(device_type)

  def testBasicCopyToMesh(self):
    target_layout = Layout.replicated(self.first_mesh, rank=1)
    numpy_value = np.zeros([3], dtype=np.int32)
    dtensor_copy_from_numpy = api.copy_to_mesh(numpy_value, target_layout)
    self.assertDTensorEqual(numpy_value, target_layout, dtensor_copy_from_numpy)

    numpy_value = np.ones([3], dtype=np.int32)
    src_mesh = api.copy_to_mesh(
        numpy_value, Layout.replicated(self.second_mesh, rank=1)
    )
    dtensor_from_another_mesh = api.copy_to_mesh(src_mesh, target_layout)
    self.assertDTensorEqual(
        numpy_value, target_layout, dtensor_from_another_mesh
    )

  @parameterized.named_parameters(
      dict(testcase_name='Graph', is_eager=False),
      dict(testcase_name='Eager', is_eager=True),
  )
  def testCopyToMeshOneToOneSharded(self, is_eager):
    if not test_util.is_tpu_present():
      self.skipForDeviceType(
          ['CPU'], 'Need at least one device mesh for this test.'
      )

    replicated_layout = Layout.replicated(self.first_mesh, rank=1)
    first_layout = Layout([_MESH_DIM_X], self.first_mesh)
    second_layout = Layout([_MESH_DIM_X], self.second_mesh)

    numpy_value = np.zeros([8], dtype=np.int32)
    dt_value = api.copy_to_mesh(numpy_value, replicated_layout)
    self.assertDTensorEqual(numpy_value, replicated_layout, dt_value)

    def fn(val):
      dt_source = api.relayout(val, first_layout)
      dt_target = api.copy_to_mesh(dt_source, second_layout)
      return dt_source, dt_target

    if not is_eager:
      fn = polymorphic_function.function(fn)

    dt_source, dt_target = fn(dt_value)
    self.assertDTensorEqual(numpy_value, first_layout, dt_source)
    self.assertDTensorEqual(numpy_value, second_layout, dt_target)

  @parameterized.named_parameters(
      dict(testcase_name='Graph', is_eager=False),
      dict(testcase_name='Eager', is_eager=True),
  )
  def testCopyToMeshToShardedLayout(self, is_eager):
    target_layout = Layout([_MESH_DIM_X], self.first_mesh)
    a_np = array_ops.zeros([8], dtype=dtypes.int32)

    def fn(val):
      return api.copy_to_mesh(val, target_layout)

    if not is_eager:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.first_mesh):
      dt_value = fn(a_np)

    self.assertDTensorEqual(a_np, target_layout, dt_value)

  def testNestedDefaultMesh(self):

    @polymorphic_function.function
    def func(a):
      return a + 3.0

    with api.default_mesh(self.first_mesh):
      with api.default_mesh(self.second_mesh):
        with api.default_mesh(self.first_mesh):
          result = func(array_ops.ones(shape=()))
          self.assertEqual(api.fetch_layout(result).mesh, self.first_mesh)
        result = func(array_ops.ones(shape=()))
        self.assertEqual(api.fetch_layout(result).mesh, self.second_mesh)
      result = func(array_ops.ones(shape=()))
      self.assertEqual(api.fetch_layout(result).mesh, self.first_mesh)

  def testImplicitCopyToCPUMeshForStrings(self):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=0)
    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=0)

    string_tensor = constant_op.constant('hello')

    @polymorphic_function.function
    def f(tensor, dtensor_a, dtensor_b):
      # Return an identity op for all three inputs so that linter does not
      # complain about unused variables.
      return tensor, dtensor_a, dtensor_b

    cpu_dtensor = api.copy_to_mesh(
        constant_op.constant(1), replicated_layout_on_cpu
    )
    tpu_dtensor = api.copy_to_mesh(
        constant_op.constant(1), replicated_layout_on_tpu
    )

    string_dtensor, _, _ = f(string_tensor, cpu_dtensor, tpu_dtensor)

    # Regular string tensor should be implicitly copied onto the CPU mesh,
    # not the TPU mesh.
    self.assertEqual(api.fetch_layout(string_dtensor), replicated_layout_on_cpu)

  def testMultiMeshBroadcast(self):
    first_mesh_a = api.copy_to_mesh(
        np.zeros([3], dtype=np.int32),
        Layout.replicated(self.first_mesh, rank=1),
    )
    second_mesh_a = api.copy_to_mesh(
        np.ones([3], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1),
    )
    self.assertDTensorEqual(
        np.asarray([1, 1, 1], dtype=np.int32),
        Layout.replicated(self.first_mesh, rank=1), first_mesh_a + 1)
    # Run an add with small constant - the constant should be broadcasted
    # onto the second mesh rather than the first.
    self.assertDTensorEqual(
        np.asarray([2, 2, 2], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1), second_mesh_a + 1)

  def testMultiMeshAdd(self):
    a = constant_op.constant(1, dtype=dtypes.int32)
    b = constant_op.constant(2, dtype=dtypes.int32)
    with ops.device_v2(api.device_name()):
      first_mesh_a = api.copy_to_mesh(
          a, Layout.replicated(self.first_mesh, rank=0)
      )
      first_mesh_b = api.copy_to_mesh(
          b, Layout.replicated(self.first_mesh, rank=0)
      )
    # Copy-to-mesh doesn't work with multi-mesh as we always broadcast to
    # default mesh.
    # TODO(hthu): Use copy-to-mesh after the generic Relayout CL is in.
    second_mesh_a = api.copy_to_mesh(
        np.ones([3], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1),
    )
    second_mesh_b = api.copy_to_mesh(
        np.zeros([3], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1),
    )

    first_mesh_result = first_mesh_a + first_mesh_b
    second_mesh_result = second_mesh_a + second_mesh_b
    self.assertDTensorEqual(
        np.asarray(3, dtype=np.int32),
        Layout.replicated(self.first_mesh, rank=0), first_mesh_result)
    self.assertDTensorEqual(
        np.ones([3], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1), second_mesh_result)

  def testMultiMeshFunc(self):
    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    with ops.device_v2(api.device_name()):
      first_mesh_a = api.copy_to_mesh(
          a, Layout.replicated(self.first_mesh, rank=1)
      )
    second_mesh_a = api.copy_to_mesh(
        np.ones([4], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1),
    )
    with self.assertRaises(errors_impl.UnknownError):
      # fails mesh propagation as it requires all inputs to be on the same
      # mesh.
      # pylint: disable=pointless-statement
      first_mesh_a + second_mesh_a
      # pylint: enable=pointless-statement

  def testMultiMeshInSideFunctionLayoutV2(self):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=1)
    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=0)

    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)

    def func(t):
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      return math_ops.sqrt(t)

    golden_result = func(a)

    a = api.copy_to_mesh(a, replicated_layout_on_tpu)

    @polymorphic_function.function
    def cpu_func(t):
      return math_ops.sqrt(t)

    @polymorphic_function.function
    def tpu_func(t):
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      cpu_tensor = api.copy_to_mesh(t, replicated_layout_on_cpu)
      return cpu_func(cpu_tensor)

    with ops.device_v2(api.device_name()):
      output = tpu_func(a)
      self.assertDTensorEqual(golden_result, replicated_layout_on_cpu, output)

  def testMultiMeshCancellation(self):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    replicated_layout_on_cpu = Layout([UNSHARDED], host_cpu_mesh)
    replicated_layout_on_tpu = Layout([UNSHARDED], self.second_mesh)

    @polymorphic_function.function
    def cpu_func(x):
      # Integer division by 0, which returns a bad status.
      x = math_ops.cast(gen_math_ops.div(x=x, y=x), dtypes.float32)
      return math_ops.cast(x, dtypes.float32)

    @polymorphic_function.function
    def tpu_func(cpu_tensor):
      cpu_result = cpu_func(cpu_tensor)
      tpu_tensor = api.copy_to_mesh(cpu_result, replicated_layout_on_tpu)
      # A reduction on the TPU mesh which must be cancelled in response to the
      # CPU mesh's failure.
      return math_ops.reduce_sum(tpu_tensor)

    cpu_tensor = api.copy_to_mesh(
        constant_op.constant([0, 1]), replicated_layout_on_cpu
    )

    with self.assertRaisesRegex(Exception, 'Integer division by zero'):
      # Ensure any errors are raised at end of scope.
      with context.async_scope():
        with ops.device_v2(api.device_name()):
          tpu_func(cpu_tensor)

  def testMultiMeshCPUToTPUTransfer(self):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    multiple_host_device_id = test_util.create_device_ids_array((2,))
    host_multi_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        multiple_host_device_id,
        np.ravel(multiple_host_device_id).tolist(),
        test_util.create_device_list((2,), 'CPU'),
    )

    replicated_layout_on_cpu = Layout.replicated(host_multi_cpu_mesh, rank=1)
    sharded_layout_on_tpu_r1 = Layout([_MESH_DIM_X], self.second_mesh)
    replicated_layout_on_tpu_r1 = Layout.replicated(self.second_mesh, rank=1)

    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    a = api.copy_to_mesh(a, replicated_layout_on_cpu)

    @polymorphic_function.function
    def tpu_func(t):
      return api.relayout(t, sharded_layout_on_tpu_r1)

    @polymorphic_function.function
    def cpu_func(t):
      t = math_ops.cast(t, dtypes.float32)
      tpu_tensor = api.copy_to_mesh(t, replicated_layout_on_tpu_r1)
      return tpu_func(tpu_tensor)

    with ops.device_v2(api.device_name()):
      output = cpu_func(a)

    api.check_layout(output, sharded_layout_on_tpu_r1)

  def testMultiMeshUnsupportedTypes(self):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )

    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=1)
    replicated_layout_on_tpu_r1 = Layout.replicated(self.second_mesh, rank=1)

    s = constant_op.constant(['a', 'b', 'c'], dtype=dtypes.string)
    s = api.copy_to_mesh(s, replicated_layout_on_cpu)

    @polymorphic_function.function
    def tpu_func(t):
      return array_ops.identity(t)

    @polymorphic_function.function
    def cpu_func(t):
      t = array_ops.identity(t)
      tpu_tensor = api.copy_to_mesh(t, replicated_layout_on_tpu_r1)
      return tpu_func(tpu_tensor)

    with self.assertRaises(errors_impl.UnknownError) as ex:
      with ops.device_v2(api.device_name()):
        _ = str(cpu_func(s))

    self.assertIn('unsupported output type', ex.exception.message)

  def testMultiMeshCPUToCPUTransfer(self):
    send_device_id = test_util.create_device_ids_array((1,))
    send_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        send_device_id,
        np.ravel(send_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    recv_cpu_mesh = Mesh.from_string(
        '|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:1'
    )

    replicated_layout_on_cpu_send = Layout.replicated(send_cpu_mesh, rank=1)
    replicated_layout_on_cpu_recv = Layout.replicated(recv_cpu_mesh, rank=1)
    replicated_layout_on_cpu_r0 = Layout.replicated(recv_cpu_mesh, rank=0)

    def func(t):
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      return math_ops.sqrt(t)

    @polymorphic_function.function
    def cpu_recv_func(t):
      t = math_ops.reduce_sum(t)
      t = math_ops.sqrt(t)
      return t

    @polymorphic_function.function
    def cpu_send_func(t):
      t = math_ops.cast(t, dtypes.float32)
      cpu_recv_tensor = api.copy_to_mesh(t, replicated_layout_on_cpu_recv)
      t = cpu_recv_func(cpu_recv_tensor)
      return t

    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    golden_result = func(a)
    a = api.copy_to_mesh(a, replicated_layout_on_cpu_send)

    with ops.device_v2(api.device_name()):
      output = cpu_send_func(a)
      self.assertDTensorEqual(golden_result, replicated_layout_on_cpu_r0,
                              output)

  def testMultiMeshCPUTest(self):
    device_ids = test_util.create_device_ids_array((2,))
    cpu_mesh_a = Mesh(
        ['x'],
        device_ids,
        np.ravel(device_ids).tolist(),
        test_util.create_device_list((2,), 'CPU'),
    )
    cpu_mesh_b = Mesh(
        ['y'],
        device_ids,
        np.ravel(device_ids).tolist(),
        test_util.create_device_list((2,), 'CPU'),
    )
    replicated_layout_on_a = Layout.replicated(cpu_mesh_a, rank=1)
    replicated_layout_on_b = Layout.replicated(cpu_mesh_b, rank=1)

    x = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    y = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)

    a = api.copy_to_mesh(x, replicated_layout_on_a)
    b = api.copy_to_mesh(y, replicated_layout_on_b)

    @polymorphic_function.function
    def func2(t1, t2):
      t1 = math_ops.cast(t1, dtypes.float32)
      t1 = t1 * t1

      t2 = math_ops.cast(t2, dtypes.float32)
      t2 = math_ops.sqrt(t2)
      return t1, t2

    with ops.device_v2(api.device_name()):
      output1, output2 = func2(a, b)

    api.check_layout(output1, replicated_layout_on_a)
    api.check_layout(output2, replicated_layout_on_b)

  def testFunctionWithMultiMeshInputOutputs(self):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=1)
    replicated_layout_on_cpu_r0 = Layout.replicated(host_cpu_mesh, rank=0)
    replicated_layout_on_tpu_r0 = Layout.replicated(self.second_mesh, rank=0)
    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=1)

    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    b = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)

    def golden_func(t1, t2):
      t1 = math_ops.cast(t1, dtypes.float32)
      t1 = t1 * t1
      t2 = math_ops.cast(t2, dtypes.float32)
      t2 = math_ops.reduce_sum(t2)
      out1 = gen_math_ops.neg(t2)

      t2 = t2 + t1
      out0 = math_ops.sqrt(t2)
      return out0, out1

    golden_result0, golden_result1 = golden_func(a, b)

    cpu_dtensor = api.copy_to_mesh(a, replicated_layout_on_cpu)
    tpu_dtensor = api.copy_to_mesh(b, replicated_layout_on_tpu)

    @polymorphic_function.function
    def cpu_func(t1, t2):
      t2 = t2 + t1
      return math_ops.sqrt(t2)

    @polymorphic_function.function
    def func(tpu_input, cpu_input):
      cpu_input = math_ops.cast(cpu_input, dtypes.float32)
      cpu_input = cpu_input * cpu_input
      tpu_input = math_ops.cast(tpu_input, dtypes.float32)
      tpu_input = math_ops.reduce_sum(tpu_input)
      tpu_output = gen_math_ops.neg(tpu_input)

      cpu_tensor = api.copy_to_mesh(tpu_input, replicated_layout_on_cpu_r0)
      cpu_output = cpu_func(cpu_tensor, cpu_input)
      return cpu_output, tpu_output

    with ops.device_v2(api.device_name()):
      output0, output1 = func(tpu_dtensor, cpu_dtensor)

    self.assertDTensorEqual(golden_result0, replicated_layout_on_cpu, output0)
    self.assertDTensorEqual(golden_result1, replicated_layout_on_tpu_r0,
                            output1)

  def testMultiMeshWithResourceOps(self):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )

    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=0)
    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=1)
    a = constant_op.constant(
        [1, 2, 3, 4], dtype=dtypes.int64
    )  # NOTE(b/274627284): Variable of int32 type on GPU doesn't work.

    def func(t):
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      return math_ops.sqrt(t)

    golden_result = func(a)

    @polymorphic_function.function
    def cpu_func(t):
      return math_ops.sqrt(t)

    @polymorphic_function.function
    def tpu_func(t):
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      cpu_tensor = api.copy_to_mesh(t, replicated_layout_on_cpu)
      return cpu_func(cpu_tensor)

    with ops.device_v2(api.device_name()):
      v = api.copy_to_mesh(a, replicated_layout_on_tpu)
      w = d_variable.DVariable(v)
      output = tpu_func(w)

    self.assertDTensorEqual(golden_result, replicated_layout_on_cpu, output)

  @parameterized.named_parameters(
      ('_host_to_dev_sharded_i32', True, True, dtypes.int32),
      ('_dev_to_host_sharded_i32', False, True, dtypes.int32),
      ('_host_to_dev_replicated_i32', True, False, dtypes.int32),
      ('_dev_to_host_replicated_i32', False, False, dtypes.int32),
      ('_host_to_dev_sharded_bf16', True, True, dtypes.bfloat16),
      ('_dev_to_host_sharded_bf16', False, True, dtypes.bfloat16),
      ('_host_to_dev_replicated_bf16', True, False, dtypes.bfloat16),
      ('_dev_to_host_replicated_bf16', False, False, dtypes.bfloat16),
      ('_host_to_dev_sharded_f32', True, True, dtypes.float32),
      ('_dev_to_host_sharded_f32', False, True, dtypes.float32),
      ('_host_to_dev_replicated_f32', True, False, dtypes.float32),
      ('_dev_to_host_replicated_f32', False, False, dtypes.float32),
      ('_host_to_dev_sharded_f64', True, True, dtypes.float64),
      ('_dev_to_host_sharded_f64', False, True, dtypes.float64),
      ('_host_to_dev_replicated_f64', True, False, dtypes.float64),
      ('_dev_to_host_replicated_f64', False, False, dtypes.float64),
  )
  def testMultiMeshHostDeviceTransfer(self, host_to_dev, sharded, dtype):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    def run_copy_to_mesh(data, src_layout, dst_layout):

      @polymorphic_function.function
      def func(x):
        return api.copy_to_mesh(x, dst_layout)

      if src_layout.is_fully_replicated():
        src_data = api.copy_to_mesh(data, src_layout)
      else:
        src_data = api.copy_to_mesh(
            data, Layout.replicated(src_layout.mesh, rank=len(data.shape))
        )
        src_data = api.relayout(src_data, src_layout)
      dst_data = func(src_data)
      return (src_data, dst_data)

    dev_mesh = self.first_mesh
    cpu_mesh = self.second_mesh

    if host_to_dev:
      src_mesh, dst_mesh = cpu_mesh, dev_mesh
    else:
      src_mesh, dst_mesh = dev_mesh, cpu_mesh

    if sharded:
      src_layout = Layout.batch_sharded(src_mesh, src_mesh.dim_names[0], rank=2)
      dst_layout = Layout.batch_sharded(dst_mesh, dst_mesh.dim_names[0], rank=2)
    else:
      src_layout = Layout.replicated(src_mesh, rank=2)
      dst_layout = Layout.replicated(dst_mesh, rank=2)

    data = array_ops.ones([8, 8], dtype=dtype)
    src, dst = run_copy_to_mesh(data, src_layout, dst_layout)
    self.assertDTensorEqual(data, src_layout, src)
    self.assertDTensorEqual(data, dst_layout, dst)

  @parameterized.named_parameters(('_host_to_tpu', True),
                                  ('_tpu_to_host', False))
  def testMultiMeshWithHostMesh(self, host_to_tpu):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    sharded_layout_on_tpu = Layout([_MESH_DIM_X], self.second_mesh)
    host_layout = Layout(sharded_layout_on_tpu.sharding_specs,
                         sharded_layout_on_tpu.mesh.host_mesh())

    if host_to_tpu:
      source_layout = host_layout
      target_layout = sharded_layout_on_tpu
    else:
      source_layout = sharded_layout_on_tpu
      target_layout = host_layout

    numpy_a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)

    # TODO(b/193443769): switch to a single copy_to_mesh when this is supported.
    replicated_layout = Layout.replicated(source_layout.mesh,
                                          source_layout.rank)
    a = api.copy_to_mesh(numpy_a, replicated_layout)
    a = api.relayout(a, source_layout)

    @polymorphic_function.function
    def func(t):
      target_tensor = api.copy_to_mesh(t, target_layout)
      return array_ops.identity(target_tensor)

    with ops.device_v2(api.device_name()):
      dtensor_output = func(a)

    self.assertDTensorEqual(numpy_a, target_layout, dtensor_output)

  def testMultiMeshBackward(self):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=1)
    host_layout = Layout.replicated(self.second_mesh.host_mesh(), rank=1)

    source_layout = host_layout
    target_layout = replicated_layout_on_tpu

    @polymorphic_function.function
    def func(x):
      with backprop.GradientTape() as tape:
        tape.watch(x)
        x = x * 4.0
        t = api.copy_to_mesh(x, target_layout)
        sqrt = math_ops.sqrt(t)
        sqrt_grad = tape.gradient(sqrt, x)
        return sqrt_grad

    @polymorphic_function.function
    def second(x):
      with backprop.GradientTape() as tape:
        tape.watch(x)
        sqrt_grad = func(x)
        sqrt_grad_grad = tape.gradient(sqrt_grad, x)
        return sqrt_grad_grad

    numpy_a = constant_op.constant([1, 4, 16, 64], dtype=dtypes.float32)
    a = api.copy_to_mesh(numpy_a, source_layout)

    with ops.device_v2(api.device_name()):
      a_grad = func(a)

    self.assertDTensorEqual(0.5 * 0.5 * (1 / numpy_a)**0.5, host_layout, a_grad)
    with ops.device_v2(api.device_name()):
      a_grad_grad = second(a)
    self.assertDTensorEqual(-0.5 * 0.5 * 0.5 * (1 / numpy_a)**1.5, host_layout,
                            a_grad_grad)

  def testMultiMeshMultipleCopyToMesh(self):
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    sharded_layout_on_tpu = Layout([_MESH_DIM_X], self.second_mesh)
    host_layout = Layout(
        sharded_layout_on_tpu.sharding_specs,
        sharded_layout_on_tpu.mesh.host_mesh(),
    )

    source_layout = host_layout
    target_layout = sharded_layout_on_tpu

    numpy_a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    numpy_b = constant_op.constant([2, 2, 3, 4], dtype=dtypes.int32)

    # TODO(b/193443769): switch to a single copy_to_mesh when this is supported.
    replicated_layout = Layout.replicated(
        source_layout.mesh, source_layout.rank
    )
    a = api.copy_to_mesh(numpy_a, replicated_layout)
    b = api.copy_to_mesh(numpy_b, replicated_layout)
    a = api.relayout(a, source_layout)
    b = api.relayout(b, source_layout)

    @polymorphic_function.function
    def func(a, b):
      a = api.copy_to_mesh(a, target_layout)
      b = api.copy_to_mesh(b, target_layout)
      return array_ops.identity(a), array_ops.identity(b)

    with ops.device_v2(api.device_name()):
      dtensor_a, dtensor_b = func(a, b)

    self.assertDTensorEqual(numpy_a, target_layout, dtensor_a)
    self.assertDTensorEqual(numpy_b, target_layout, dtensor_b)

  def testDVariableDefaultMesh(self):
    other_layout = Layout.replicated(_OTHER_CPU_MESH, rank=0)
    first_layout = Layout.replicated(_ONE_D_CPU_MESH, rank=0)

    _ = api.copy_to_mesh(1.0, other_layout)
    init_value = api.copy_to_mesh(1.0, first_layout)
    _ = d_variable.DVariable(init_value)


if __name__ == '__main__':
  test.main()
