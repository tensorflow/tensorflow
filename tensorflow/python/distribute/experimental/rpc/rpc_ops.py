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
"""Module to expose RPC APIs in tensorflow."""

from typing import Any, Callable, Optional, Sequence, Union

import tensorflow.distribute.experimental.rpc.kernels.gen_rpc_ops as gen_rpc_ops
from tensorflow.distribute.experimental.rpc.proto import tf_rpc_service_pb2 as rpc_pb2
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import nest


def get_output_specs_from_function(func: tf_function.ConcreteFunction):
  output_specs = nest.map_structure(type_spec.type_spec_from_value,
                                    func.structured_outputs)
  encoder = nested_structure_coder.StructureCoder()
  output_specs_proto = encoder.encode_structure(output_specs)
  return output_specs_proto.SerializeToString()


def get_input_specs_from_function(func: tf_function.ConcreteFunction):
  arg_specs, _ = func.structured_input_signature
  encoder = nested_structure_coder.StructureCoder()
  arg_specs_proto = encoder.encode_structure(arg_specs)
  return arg_specs_proto.SerializeToString()


class Server(object):
  """Server object encapsulates a resource with GRPC server.

    Functions can be registered locally and are exposed via RPCs.
    Example:
    ```
    server = rpc_ops.Server("host:port")
    @tf.function
    def add(a, b):
      return a + b

    server.register("add", add)
    server.start()
    ```
  """

  def __init__(self, address: str):
    self._server_handle = gen_rpc_ops.rpc_server(address)
    if context.executing_eagerly():
      self._handle_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._server_handle, handle_device=self._server_handle.device)
    else:
      raise NotImplementedError("Please create the server outside tf.function.")

  def register(self, method_name: str,
               func: Union[def_function.Function, tf_function.ConcreteFunction,
                           Callable[..., Any]]):
    """Method for registering functions."""

    if isinstance(func, def_function.Function):
      if func._function_spec.arg_names:  # pylint: disable=protected-access
        if func.input_signature is None:
          raise ValueError("Input signature not specified for the function.")
      concrete_fn = func.get_concrete_function()
      gen_rpc_ops.rpc_server_register(
          self._server_handle,
          method_name=method_name,
          captured_inputs=concrete_fn.captured_inputs,
          input_specs=get_input_specs_from_function(concrete_fn),
          output_specs=get_output_specs_from_function(concrete_fn),
          f=concrete_fn)
    elif isinstance(func, tf_function.ConcreteFunction):
      gen_rpc_ops.rpc_server_register(
          self._server_handle,
          method_name=method_name,
          captured_inputs=func.captured_inputs,
          input_specs=get_input_specs_from_function(concrete_fn),
          output_specs=get_output_specs_from_function(func),
          f=func)
    else:
      # Python functions
      # TODO(b/186762191): Add an implementation to support python functions.
      raise ValueError("Only TF functions are supported with Register method")

  def start(self):
    """Starts GRPC server."""
    gen_rpc_ops.rpc_server_start(self._server_handle)


class Client():
  """Client wrapper to connect to remote RPC server.

  If Client is created with (list_registered_methods=True):
  1. Input and output specs for the methods till this point will be fetched from
  Server.
  2. convenience methods are added to invoke registered methods directly from
  client.
  For example:
    For call a server method `add`
    client.add(a, b) or client.add_async(a, b) can be used instead of
    client.call(args=[a,b], output_specs=[..])

  Prerequiste for using list_registered_methods=True:
   1. Server should be already started with the registered methods.
   2. Client must be created in Eager mode.
  """

  def __init__(self,
               address: str,
               name: str = "",
               list_registered_methods=False,
               timeout_in_ms=0):
    self._client_handle, methods = gen_rpc_ops.rpc_client(
        shared_name=name,
        server_address=address,
        list_registered_methods=list_registered_methods,
        timeout_in_ms=timeout_in_ms)
    if context.executing_eagerly():
      self._handle_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._client_handle, handle_device=self._client_handle.device)
    else:
      raise NotImplementedError(
          "Client creation is supported only in eager mode.")
    self._server_address = address
    decoder = nested_structure_coder.StructureCoder()
    self._method_registry = {}
    for method in methods.numpy():

      m = rpc_pb2.RegisteredMethod()
      m.ParseFromString(method)
      output_specs = decoder.decode_proto(m.output_specs)
      input_specs = decoder.decode_proto(m.input_specs)
      self._method_registry[m.method] = output_specs
      # TODO(ishark): Perhaps doc string can also be taken as input during
      # function registration.
      doc_string = "RPC Call for " + m.method + " method to server " + address
      self._add_method(m.method, output_specs, input_specs, self._client_handle,
                       doc_string)

  def _add_method(self, method_name, output_specs, input_specs, client_handle,
                  doc_string):
    """Method to add RPC methods to the client object."""

    def validate_and_get_flat_inputs(*args):
      if args is None:
        args = []
      if input_specs:
        nest.assert_same_structure(args, input_specs)
      flat_inputs = nest.flatten(args)
      return flat_inputs

    def call_wrapper(*args):
      status_or, deleter = gen_rpc_ops.rpc_call(
          client_handle,
          args=validate_and_get_flat_inputs(*args),
          method_name=method_name)
      return StatusOrResult(status_or, deleter, output_specs)

    setattr(self, method_name, call_wrapper)
    setattr(getattr(self, method_name), "__doc__", doc_string)

  def call(self,
           method_name: str,
           args: Optional[Sequence[core_tf_types.Tensor]] = None,
           output_specs=None):
    """Method to invoke remote registered functions on the connected server.

    Server should be started before making an RPC Call.

    Args:
      method_name: Registered method to invoke on Server.
      args: Input arguments for the method.
      output_specs: Output specs for the output from method.
       For example, if tf function is:
         @tf.function(input_signature=[
            tensor_spec.TensorSpec([], tf.int32),
            tensor_spec.TensorSpec([], tf.int32)
        ])
        def multiply_fn(a, b):
          return tf.math.multiply(a, b)

       output_spec is: tf.TensorSpec((), tf.int32)

       If you have access to TF Function, the output specs can be generated
       from tf.function by calling:
         output_specs = tf.nest.map_structure(tf.type_spec_from_value,
                  tf_function.get_concrete_function().structured_outputs)

    Returns:
      StatusOrResult object. This function issues the RPC call to server, it
      does not block for the duration of RPC. Please call is_ok, get_error or
      get_value methods on the returned object to blocked till RPC finishes.
    """
    if args is None:
      args = []
    status_or, deleter = gen_rpc_ops.rpc_call(
        self._client_handle, args=nest.flatten(args), method_name=method_name)
    return StatusOrResult(status_or, deleter, output_specs)


class StatusOrResult(object):
  """Class representing result and status from RPC Call."""

  def __init__(self, status_or, deleter, output_specs=None):
    self._status_or = status_or
    self._output_specs = output_specs
    self._deleter = deleter
    self._error_code, self._error_message = None, None

  def _check_status(self):
    if self._error_code is None:
      self._error_code, self._error_message = gen_rpc_ops.rpc_check_status(
          self._status_or)

  def __del__(self):
    # Make sure the resource is deleted in the same mode as it was created in.
    if context.executing_eagerly():
      with context.eager_mode():
        gen_rpc_ops.delete_rpc_future_resource(
            handle=self._status_or, deleter=self._deleter)
    else:
      with context.graph_mode():
        gen_rpc_ops.delete_rpc_future_resource(
            handle=self._status_or, deleter=self._deleter)

  def is_ok(self):
    self._check_status()
    return math_ops.equal(self._error_code,
                          constant_op.constant(0, dtype=dtypes.int64))

  def get_error(self):
    self._check_status()
    return self._error_code, self._error_message

  def get_value(self):
    """output_specs: Output specs for the output from method.

    For example, if tf function is:
       @tf.function(input_signature=[
          tensor_spec.TensorSpec([], tf.int32),
          tensor_spec.TensorSpec([], tf.int32)
        ])
      def multiply_fn(a, b):
        return tf.math.multiply(a, b)

     output_spec is: tf.TensorSpec((), tf.int32)

     If you have access to TF Function, the output specs can be generated
     from tf.function by calling:
       output_specs = tf.nest.map_structure(tf.type_spec_from_value,
                tf_function.get_concrete_function().structured_outputs)


    Returns:
    Output of the RPC call.
    """

    self._check_status()
    if self._output_specs is None or isinstance(self._output_specs,
                                                structure.NoneTensorSpec):
      flat_output_dtypes = []
      return_none = True
    else:
      return_none = False
      flat_output_dtypes = [s.dtype for s in nest.flatten(self._output_specs)]

    result = gen_rpc_ops.rpc_get_value(self._status_or, Tout=flat_output_dtypes)
    if return_none:
      return None
    else:
      return nest.pack_sequence_as(self._output_specs, result)
