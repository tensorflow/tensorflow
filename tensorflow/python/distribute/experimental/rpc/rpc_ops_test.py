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
"""Tests for rpc_ops.py."""

import threading
import time
import numpy as np
import portpicker

from tensorflow.python.distribute.experimental.rpc import rpc_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as eager_def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class RpcOpsTest(test.TestCase):

  def setUp(self):
    super(RpcOpsTest, self).setUp()
    cpus = config.list_physical_devices("CPU")
    # Set 2 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

  def test_generated_rpc_ops(self):
    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def remote_fn(a, b):
      return math_ops.multiply(a, b)

    concrete_remote_fn = remote_fn.get_concrete_function()

    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.gen_rpc_ops.rpc_server(server_address=address)

    rpc_ops.gen_rpc_ops.rpc_server_register(
        server_resource,
        f=concrete_remote_fn,
        captured_inputs=concrete_remote_fn.captured_inputs,
        output_specs=rpc_ops.get_output_specs_from_function(concrete_remote_fn),
        method_name="multiply")

    rpc_ops.gen_rpc_ops.rpc_server_start(server_resource)
    client_handle, _ = rpc_ops.gen_rpc_ops.rpc_client(
        server_address=address, timeout_in_ms=5000)
    future_resource, deleter = rpc_ops.gen_rpc_ops.rpc_call(
        client_handle, args=[a, b], method_name="multiply", timeout_in_ms=0)

    error_code, _ = rpc_ops.gen_rpc_ops.rpc_check_status(future_resource)
    self.assertAllEqual(error_code, 0)
    self.assertAllEqual(
        rpc_ops.gen_rpc_ops.rpc_get_value(future_resource, Tout=[dtypes.int32]),
        [6])

    resource_variable_ops.EagerResourceDeleter(
        handle=server_resource, handle_device=server_resource.device)

    resource_variable_ops.EagerResourceDeleter(
        handle=client_handle, handle_device=client_handle.device)

    rpc_ops.gen_rpc_ops.delete_rpc_future_resource(future_resource, deleter)

  def test_exported_rpc_api_static_factory(self):

    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def _remote_fn(a, b):
      return math_ops.multiply(a, b)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.Server.create("grpc", address)
    server_resource.register("multiply", _remote_fn)

    server_resource.start()
    client = rpc_ops.Client.create("grpc", address=address, name="test_client")

    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)

    mul_or = client.call(
        args=[a, b],
        method_name="multiply",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))

    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    # Test empty client name
    client1 = rpc_ops.Client.create("grpc", address)
    mul_or = client1.call(
        args=[a, b],
        method_name="multiply",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))
    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    # Test without output_spec
    mul_or = client1.multiply(a, b)
    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    self.assertEqual(client1.multiply.__doc__,
                     "RPC Call for multiply method to server " + address)

  def test_rpc_ops_wrapper(self):

    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def _remote_fn(a, b):
      return math_ops.multiply(a, b)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def add_fn(a, b):
      return math_ops.add(a, b)

    # Register TF function
    server_resource.register("multiply", _remote_fn)

    # Register concrete Function
    server_resource.register("add", add_fn.get_concrete_function())

    server_resource.start()
    client = rpc_ops.GrpcClient(address=address, name="test_client")

    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)

    mul_or = client.call(
        args=[a, b],
        method_name="multiply",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))

    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    add_or = client.call(
        args=[a, b],
        method_name="add",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))

    self.assertAllEqual(add_or.is_ok(), True)
    self.assertAllEqual(add_or.get_value(), 5)

    # Test empty client name
    client1 = rpc_ops.GrpcClient(address, list_registered_methods=True)
    mul_or = client1.call(
        args=[a, b],
        method_name="multiply",
        output_specs=tensor_spec.TensorSpec((), dtypes.int32))
    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    # Test without output_spec
    mul_or = client1.multiply(a, b)
    self.assertAllEqual(mul_or.is_ok(), True)
    self.assertAllEqual(mul_or.get_value(), 6)

    self.assertEqual(client1.multiply.__doc__,
                     "RPC Call for multiply method to server " + address)

  def test_output_specs(self):

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def test_dict(val):
      return {"key": val}

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def is_positive(a):
      if a > 0:
        return True
      return False

    @eager_def_function.function(input_signature=[])
    def do_nothing():
      return []

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def test_nested_structure(v):
      return {"test": (v, [v, v]), "test1": (v,)}

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    server_resource.register("test_dict", test_dict)
    server_resource.register("is_positive", is_positive)
    server_resource.register("test_nested_structure", test_nested_structure)
    server_resource.register("do_nothing", do_nothing)

    server_resource.start()

    client = rpc_ops.GrpcClient(
        address=address, name="test_client", list_registered_methods=True)

    a = variables.Variable(2, dtype=dtypes.int32)

    result_or = client.test_dict(a)
    self.assertAllEqual(result_or.is_ok(), True)
    nest.map_structure(self.assertAllEqual, result_or.get_value(), {"key": 2})

    self.assertTrue(client.is_positive(a))

    result_or = client.test_nested_structure(a)
    self.assertAllEqual(result_or.is_ok(), True)
    nest.map_structure(self.assertAllEqual, result_or.get_value(), {
        "test": (2, [2, 2]),
        "test1": (2,)
    })

    result_or = client.do_nothing()
    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), [])

  def test_input_specs(self):

    @eager_def_function.function(input_signature=[{
        "a": tensor_spec.TensorSpec([], dtypes.int32),
        "b": tensor_spec.TensorSpec([], dtypes.int32)
    }])
    def test_input_dict(value):
      return math_ops.add(value["a"], value["b"])

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    server_resource.register("test_input_dict", test_input_dict)

    server_resource.start()

    client = rpc_ops.GrpcClient(
        address=address, name="test_client", list_registered_methods=True)
    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)
    result_or = client.test_input_dict({"a": a, "b": b})
    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), 5)

    with self.assertRaises(TypeError):
      client.test_input_dict([a, b])

  def test_call_register_ordering(self):
    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)

    # Create client succeeds before server start and registration
    client = rpc_ops.GrpcClient(address)

    # Create client with list_registered_methods fails before server is started.
    with self.assertRaises(errors.DeadlineExceededError):
      rpc_ops.GrpcClient(
          address,
          name="client1",
          list_registered_methods=True,
          timeout_in_ms=1)

    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      v.assign_add(a)

    @eager_def_function.function(input_signature=[])
    def read_var():
      return v.value()

    server = rpc_ops.GrpcServer(address)

    def start_server():
      # Delay server start to test whether client creation also waits
      # till server is up.
      time.sleep(1)
      server.register("assign_add", assign_add)
      server.start()

    t = threading.Thread(target=start_server)
    t.start()

    # Create same "client1" again should succeed.
    client1_with_listed_methods = rpc_ops.GrpcClient(
        address, name="client1", list_registered_methods=True)

    result_or = client1_with_listed_methods.assign_add(
        variables.Variable(2, dtype=dtypes.int64))
    self.assertAllEqual(result_or.is_ok(), True)

    result_or = client.call("assign_add",
                            [variables.Variable(2, dtype=dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)

    # Create client with registered methods
    client2_with_listed_methods = rpc_ops.GrpcClient(
        address=address, name="client2", list_registered_methods=True)

    result_or = client2_with_listed_methods.assign_add(
        variables.Variable(2, dtype=dtypes.int64))
    self.assertAllEqual(result_or.is_ok(), True)

    self.assertAllEqual(v, 6)

    # Register new method after server started.
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "All methods must be registered before starting the server"):
      server.register("read_var", read_var)

  def test_client_timeout(self):
    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)

    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def add(a, b):
      return math_ops.add(a, b)

    server = rpc_ops.GrpcServer(address)

    def start_server():
      # Delay server start to simulate deadline exceeded for 1st RPC call
      # response. Client waits till server is started, thus it can trigger
      # deadline exceeded.
      time.sleep(1)
      server.register("add", add)
      server.start()

    t = threading.Thread(target=start_server)
    t.start()

    # Create client with list_registered_methods fails before server is started.
    with self.assertRaises(errors.DeadlineExceededError):
      rpc_ops.GrpcClient(
          address,
          name="client1",
          list_registered_methods=True,
          timeout_in_ms=1)

    # Create same client again should succeed with
    # list_registered_methods=False. Default timeout for client is 1 ms.
    client = rpc_ops.GrpcClient(
        address, name="client1", list_registered_methods=False, timeout_in_ms=1)

    # Make explicit RPC call, the default timeout of 1 ms should lead to
    # deadline exceeded error.
    result_or = client.call(
        "add", [constant_op.constant(20),
                constant_op.constant(30)])
    self.assertAllEqual(result_or.is_ok(), False)
    error_code, _ = result_or.get_error()
    self.assertAllEqual(error_code, errors.DEADLINE_EXCEEDED)

    # Specifying reasonable timeout for call should succeed.
    result_or = client.call(
        "add", [constant_op.constant(20),
                constant_op.constant(30)],
        timeout_in_ms=5000)
    self.assertAllEqual(result_or.is_ok(), True)
    error_code, _ = result_or.get_error()

    # Test timeouts for convenience methods

    # Client with no default timeout.
    client = rpc_ops.GrpcClient(
        address, name="client2", list_registered_methods=True)

    # Restart server again with delay to simulate deadline exceeded.
    del server
    server = rpc_ops.GrpcServer(address)
    t = threading.Thread(target=start_server)
    t.start()

    # Call fails with 1 ms timeout.
    result_or = client.add(
        constant_op.constant(20), constant_op.constant(30), timeout_in_ms=1)
    self.assertAllEqual(result_or.is_ok(), False)
    error_code, _ = result_or.get_error()
    self.assertAllEqual(error_code, errors.DEADLINE_EXCEEDED)

    # Succeeds with reasonable timeout.
    result_or = client.add(
        constant_op.constant(20), constant_op.constant(30), timeout_in_ms=5000)
    self.assertAllEqual(result_or.is_ok(), True)

  def test_async_call_op_wrapper(self):
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      v.assign_add(a)

    @eager_def_function.function(input_signature=[])
    def read_var():
      return v.value()

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("assign_add", assign_add)
    server.register("read_var", read_var)
    server.start()

    client = rpc_ops.GrpcClient(address)

    futures = []
    for _ in range(10):
      futures.append(
          client.call("assign_add",
                      [variables.Variable(2, dtype=dtypes.int64)]))

    for f in futures:
      f.is_ok()

    result_or = client.call(
        "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])

    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), [20])

  def test_rpc_call_op_in_tf_function(self):

    @eager_def_function.function(input_signature=[
        tensor_spec.TensorSpec([], dtypes.int32),
        tensor_spec.TensorSpec([], dtypes.int32)
    ])
    def _remote_fn(a, b):
      return math_ops.multiply(a, b)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server_resource = rpc_ops.GrpcServer(address)

    server_resource.register("remote_fn", _remote_fn)

    server_resource.start()
    client = rpc_ops.GrpcClient(address=address, name="test_client")

    a = variables.Variable(2, dtype=dtypes.int32)
    b = variables.Variable(3, dtype=dtypes.int32)

    @eager_def_function.function
    def call_fn():
      result_or = client.call(
          args=[a, b],
          method_name="remote_fn",
          output_specs=[tensor_spec.TensorSpec([], dtypes.int32)])

      self.assertAllEqual(True, result_or.is_ok())
      result = result_or.get_value()
      self.assertEqual(len(result), 1)  # Call returns a list(tensors)
      # TODO(ishark): Shape for output tensor is unknown currently.
      # Add attribute for capturing TensorSpec for output and enable
      # check below:
      # self.assertIsNotNone(result[0].shape.rank)
      return result

    self.assertAllEqual(call_fn(), [6])

  def test_resource_deletion(self):
    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server_handle = server._server_handle

    # Test Future resource deletion
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(input_signature=[])
    def read_var():
      return v.value()

    server.register("read_var", read_var)

    server.start()
    client = rpc_ops.GrpcClient(address)

    client_handle = client._client_handle

    # Check future resource deletion without calling get_value.
    def _create_and_delete_rpc_future():
      handle = client.call(
          "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
      return handle._status_or

    @eager_def_function.function
    def _create_and_delete_rpc_future_fn():
      handle = client.call(
          "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
      return handle._status_or

    for _ in range(2):
      handle = _create_and_delete_rpc_future()
      with self.assertRaises(errors.NotFoundError):
        resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=False)

    for _ in range(2):
      handle = _create_and_delete_rpc_future_fn()
      with self.assertRaises(errors.NotFoundError):
        resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=False)

    # Check future resource deletion with calling get_value.
    def _create_and_delete_with_future():
      handle = client.call(
          "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
      status_or_handle = handle._status_or
      handle.get_value()
      return status_or_handle

    # Check future resource deletion with calling get_value with tf.function.
    @eager_def_function.function
    def _create_and_delete_with_future_fn():
      handle = client.call(
          "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
      status_or_handle = handle._status_or
      handle.get_value()
      return status_or_handle

    for _ in range(2):
      resource_handle = _create_and_delete_with_future()
      with self.assertRaises(errors.NotFoundError):
        resource_variable_ops.destroy_resource_op(
            resource_handle, ignore_lookup_error=False)

    for _ in range(2):
      resource_handle = _create_and_delete_with_future_fn()
      with self.assertRaises(errors.NotFoundError):
        resource_variable_ops.destroy_resource_op(
            resource_handle, ignore_lookup_error=False)

    # Test server client resource gets deleted.
    del client
    with self.assertRaises(errors.NotFoundError):
      resource_variable_ops.destroy_resource_op(
          client_handle, ignore_lookup_error=False)

    # Test server server resource gets deleted.
    del server
    with self.assertRaises(errors.NotFoundError):
      resource_variable_ops.destroy_resource_op(
          server_handle, ignore_lookup_error=False)

  def test_rpc_error(self):
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      v.assign_add(a)

    @eager_def_function.function(input_signature=[])
    def read_var():
      return v.value()

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("assign_add", assign_add)
    server.register("read_var", read_var)
    server.start()

    client = rpc_ops.GrpcClient(address, list_registered_methods=True)

    # confirm it works as expected when arguments are passed.
    result_or = client.call("assign_add",
                            [variables.Variable(2, dtype=dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)
    result_or = client.call(
        "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), [2])
    result_or = client.assign_add(variables.Variable(2, dtype=dtypes.int64))
    self.assertAllEqual(True, result_or.is_ok())

    result_or = client.read_var()
    self.assertAllEqual(True, result_or.is_ok())
    self.assertAllEqual(result_or.get_value(), 4)

    # Fails with invalid argument error when no arguments are passed.
    result_or = client.call("assign_add")
    self.assertAllEqual(result_or.is_ok(), False)
    error_code, _ = result_or.get_error()
    self.assertAllEqual(error_code, errors.INVALID_ARGUMENT)

  def test_captured_inputs(self):
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      v.assign_add(a)

    @eager_def_function.function(input_signature=[])
    def read_var():
      return v.value()

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("assign_add", assign_add)
    server.register("read_var", read_var)

    server.start()

    client = rpc_ops.GrpcClient(address)

    result_or = client.call("assign_add",
                            [variables.Variable(2, dtype=dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)
    result_or = client.call("assign_add",
                            [variables.Variable(2, dtype=dtypes.int64)])
    self.assertAllEqual(result_or.is_ok(), True)
    result_or = client.call(
        "read_var", output_specs=[tensor_spec.TensorSpec([], dtypes.int64)])

    self.assertAllEqual(result_or.is_ok(), True)
    self.assertAllEqual(result_or.get_value(), [4])

  def test_register_method_twice(self):
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      v.assign_add(a)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign(a):
      v.assign(a)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("assign", assign_add)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "assign is already registered."):
      # Reusing the same error name.
      server.register("assign", assign)

  def test_tf_function_register_without_input_signature(self):
    v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function
    def assign(a):
      v.assign(a)

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    with self.assertRaisesRegex(
        ValueError, "Input signature not specified for the function."):
      server.register("assign", assign)

    # Register without input signature should work for functions without input
    # args.
    @eager_def_function.function
    def read_var():
      return v.value()

    server.register("read_var", read_var)

  def test_multi_device_resource(self):
    elements = np.random.randint(100, size=[200])

    with ops.device("/device:CPU:1"):
      queue = data_flow_ops.FIFOQueue(200, dtypes.int64, shapes=[])

    @eager_def_function.function()
    def populate_queue():
      queue.enqueue_many(elements)
      queue.close()

    with ops.device("/device:CPU:0"):
      port = portpicker.pick_unused_port()
      address = "localhost:{}".format(port)
      server = rpc_ops.GrpcServer(address)
      server.register("populate_queue", populate_queue)
      server.start()

      client = rpc_ops.GrpcClient(address, list_registered_methods=True)
      client.populate_queue()

    for e in elements:
      self.assertAllEqual(e, queue.dequeue())

  def test_queue_resource(self):
    elements = np.random.randint(100, size=[200])
    queue = data_flow_ops.FIFOQueue(200, dtypes.int64, shapes=[])

    @eager_def_function.function()
    def populate_queue():
      queue.enqueue_many(elements)
      queue.close()

    port = portpicker.pick_unused_port()
    address = "localhost:{}".format(port)
    server = rpc_ops.GrpcServer(address)
    server.register("populate_queue", populate_queue)
    server.start()

    client = rpc_ops.GrpcClient(address, list_registered_methods=True)
    client.populate_queue()

    for e in elements:
      self.assertAllEqual(e, queue.dequeue())

  def test_multi_device_resource_cpu(self):
    with ops.device("/device:cpu:1"):
      v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @eager_def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int64)])
    def assign_add(a):
      v.assign_add(a)

    with ops.device("/device:CPU:0"):
      port = portpicker.pick_unused_port()
      address = "localhost:{}".format(port)
      server = rpc_ops.GrpcServer(address)
      server.register("assign_add", assign_add)
      server.start()

      client = rpc_ops.GrpcClient(address, list_registered_methods=True)
      result_or = client.assign_add(variables.Variable(2, dtype=dtypes.int64))
      self.assertAllEqual(result_or.is_ok(), True)

    self.assertAllEqual(v, 2)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
