/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifdef TF_MLRT_OPS
#else
#define TF_MLRT_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_dialect.td"
include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.td"

def CreateOp: TensorflowMlrt_Op<"createop", []> {
  let summary = "The Fallback CreateOp";

  let description = [{
    The CreateOp creates the tensorflow::OpKernel in the fallback context.
  }];

  let arguments = (ins
    StrAttr:$node_def,
    I32Attr:$op_key
  );

  let assemblyFormat = "attr-dict";
}

def ConstOp: TensorflowMlrt_Op<"constop", []> {
  let summary = "The tf_mlrt ConstOp";

  let description = [{
    The ConstOp creates a constant tensorflow::Tensor from serialized proto.
  }];

  let arguments = (ins
    StrAttr:$tensor_proto
  );

  let results = (outs
    TFTensorType:$result
  );

  let assemblyFormat = "attr-dict";
}

def ExecuteOp : TensorflowMlrt_Op<"executeop", []> {
  let summary = "The Fallback ExecuteOp";
  let description = [{
    The ExecuteOp executes an operation on the specified device.
  }];

  let arguments = (ins
    Variadic<TFTensorType>:$args,
    StrAttr:$node_def,
    I32Attr:$op_key
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type($args, $results)";
}

def ExecuteOpWithDevice: TensorflowMlrt_Op<"executeop.device", []> {
  let summary = "The Fallback ExecuteOp with custom device";
  let description = [{
    The ExecuteOp executes an operation on the specified device using a custom device.
  }];

  let arguments = (ins
    TFDeviceType:$device,
    Variadic<TFTensorType>:$args,
    StrAttr:$node_def,
    I32Attr:$op_key
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let assemblyFormat = "`(` $device`)` `(` $args `)` attr-dict `:` functional-type($args, $results)";
}

def AsyncExecuteOp : TensorflowMlrt_Op<"async_executeop", []> {
  let summary = "The Fallback ExecuteOp for tensorflow::AsyncOpKernel";
  let description = [{
    The ExecuteOp executes an operation on the specified device asynchronously.
  }];

  let arguments = (ins
    Variadic<TFTensorType>:$args,
    StrAttr:$node_def,
    I32Attr:$op_key
  );

  let results = (outs
    Variadic<MlrtFutureType>:$results
  );

  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type($args, $results)";
}

def AsyncExecuteOpWithDevice : TensorflowMlrt_Op<"async_executeop.device", []> {
  let summary = "The Fallback ExecuteOp for tensorflow::AsyncOpKernel";
  let description = [{
    The ExecuteOp executes an operation on the specified device asynchronously.
  }];

  let arguments = (ins
    TFDeviceType:$device,
    Variadic<TFTensorType>:$args,
    StrAttr:$node_def,
    I32Attr:$op_key
  );

  let results = (outs
    Variadic<MlrtFutureType>:$results
  );

  let assemblyFormat = "`(` $device`)` `(` $args `)` attr-dict `:` functional-type($args, $results)";
}

def SetResourceOp : TensorflowMlrt_Op<"set_resource", []> {
  let summary = "Set a tensor in resource array";

  let description = [{
    Set a tensor in resource array.

    arg: the tensor to be set in the resource array.
    index: the index in the resource array
  }];

  let arguments = (ins
    TFTensorType:$arg,
    I64Attr:$index
  );

  let results = (outs);

  let assemblyFormat = "operands attr-dict";
}

def GetResourceOp : TensorflowMlrt_Op<"get_resource", []> {
  let summary = "get a tensor in resource array";

  let description = [{
    Get a tensor in resource array.

    indices: the indices in the resource array.
    results: the tensor values for the corresponding indices.
  }];

  let arguments = (ins
    I64ArrayAttr:$indices
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let assemblyFormat = "attr-dict `:` type($results)";
}

def AwaitOp: TensorflowMlrt_Op<"await", [Pure]> {
  let summary = "Await a tensor from a !mlrt.future";

  let description = [{
    Await a tensor from a !mlrt.future.

    $future: A value of type !mlrt.future. The underlying value must be a tensorflow tensor.

    $result: a tensorflow tensor.
  }];

  let arguments = (ins
    MlrtFutureType:$future
  );

  let results = (outs
    TFTensorType:$result
  );

  let assemblyFormat = "operands attr-dict";
}

def AwaitAllOp: TensorflowMlrt_Op<"await_all", [Pure]> {
  let summary = "Await tensors from a list of !mlrt.future";

  let description = [{
    Await tensors from a list of !mlrt.future.

    $futures: A list of !mlrt.future. The underlying value must be tensorflow tensors.

    $results: A list of tensorflow tensors.
  }];

  let arguments = (ins
    Variadic<MlrtFutureType>:$futures
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let assemblyFormat = "$futures attr-dict `:` type($results)";
}

def PromiseOp: TensorflowMlrt_Op<"promise", []> {
  let summary = "Set a tensor in a promise";

  let description = [{
    Set a tensor in a promise.

    $promise: A value of type !mlrt.promise. The underlying value must be a tensorflow tensor.
    $tensor: A tensorflow tensor.
  }];

  let arguments = (ins
    MlrtPromiseType:$promise,
    TFTensorType:$tensor
  );

  let assemblyFormat = "operands attr-dict";
}

def PromiseFutureOp: TensorflowMlrt_Op<"promise_future", []> {
  let summary = "Set a tensor future in a promise";

  let description = [{
    Set a tensor future in a promise.

    $promise: A value of type !mlrt.promise. The underlying value must be a tensorflow tensor.
    $future: A value of type !mlrt.future. Must represents a tensorflow tensor.
  }];

  let arguments = (ins
    MlrtPromiseType:$promise,
    MlrtFutureType:$tensor
  );

  let assemblyFormat = "operands attr-dict";
}

def PromiseReturnOp: TensorflowMlrt_Op<"promise_return", [Terminator]> {
  let summary = "Set a tensor in a promise and return from a function with no output";

  let description = [{
    Set a tensor in a promise and return from a function with no output.

    $promise: A value of type !mlrt.promise. The underlying value must be a tensorflow tensor.
    $tensor: A tensorflow tensor.
  }];

  let arguments = (ins
    MlrtPromiseType:$promise,
    TFTensorType:$tensor
  );

  let assemblyFormat = "operands attr-dict";
}

def AllocateFuturesOp: TensorflowMlrt_Op<"allocate_futures", [AttrSizedResultSegments]> {
  let summary = "Allocate futures and promsies for tensorflow tensors";

  let description = [{
    Allocate futures and promsies for tensorflow tensors.

    $num_futures: The number of futures to be allocated.

    $promises: There are $num_futures promises. promises[i] shares the state with futures[i].
    $futures: There are $num_futures futures. futures[i] shares the state with promises[i].
  }];

  let arguments = (ins
    I32Attr:$num_futures
  );

  let results = (outs
    Variadic<MlrtPromiseType>:$promises,
    Variadic<MlrtFutureType>:$futures
  );
}

def TensorToIntOp : TensorflowMlrt_Op<"tensor_to_int32", [Pure]> {
  let summary = "Cast a Tensor to int32.";
  let description = [{
    Cast a Tensor to int32.

    Example:
      %one = tf_mlrt.tensor_to_int32 %src_tenosr
  }];

  let arguments = (ins TFTensorType:$src);
  let results = (outs I32:$result);
  let assemblyFormat = "operands attr-dict";
}

def BoolToTensorOp : TensorflowMlrt_Op<"bool_to_tensor", [Pure]> {
  let summary = "Cast a boolean to a tensor.";
  let description = [{
    Cast a boolean to a tensor.

    Example:
      %one = tf_mlrt.bool_to_tensor %src_bool
  }];

  let arguments = (ins I1: $src);
  let results = (outs TFTensorType:$result);

  let assemblyFormat = "operands attr-dict";
}

def PredicateOp : TensorflowMlrt_Op<"predicate", [Pure]> {
  let summary = "Converts a fallback tensor to a bool";

  let description = [{
    Note: this kernel is used for CPU tensors.

    Converts a fallback tensor to a bool with the following rules:

    - For 0D tensors, truthiness is determined by comparing against a "zero"
      value. For numerical types it is the obvious zero. For strings it is the
      empty string.

    - For >0D tensors, truthiness is determined by looking at the number of
      elements. If has zero elements, then the result is false. Otherwise the
      result is true.

    input: a fallback tensor representing the condition.
    device: the name of the tensorflow device that is associated with the
      input fallback tensor.

    output: the converted bool.
  }];

  let arguments = (ins
    TFTensorType:$input
  );

  let results = (outs
    I1:$output
  );

  let assemblyFormat = "$input attr-dict";
}

def BatchFunctionOp : TensorflowMlrt_Op<"batch_function", [Pure]> {
  let summary = "Fallback ExecuteOp specialized for tf.BatchFunction.";

  let description = [{
    This kernel executes a variant tf.BatchFunction kernel that supports having
    the `f` attribute as a bytecode function.

    Example:
      %res = tf_mlrt.batch_function(%input, %captured_input)  {
          device = "/device:CPU:0",
          f = @batch_function,
          node_def = "..."
        } : (!tf_mlrt.tensor,!tf_mlrt.tensor) -> (!tf_mlrt.tensor)

    Note that the trailing number indicates the number of results.
  }];

  let arguments = (ins
    Variadic<TFTensorType>:$args,
    StrAttr:$device,
    SymbolRefAttr:$f,
    StrAttr:$node_def
  );

  let results = (outs
    Variadic<MlrtFutureType>:$results
  );

  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type($args, $results)";
}

def CancelOp: TensorflowMlrt_Op<"cancel", []> {
  let summary = "Handle cancellation request.";

  let description = [{
    This kernel will early terminate the program upon cancellation request (e.g. time out).
  }];
}

def MapFnOp : TensorflowMlrt_Op<"map_fn", [AttrSizedOperandSegments, Pure]> {
  let summary = "The Parallel Map for tf_mlrt dialect";
  let description = [{
    The Pmap executes body function in parallel for all ranges up to $max_iterations.

    The pseudo code:
      for(int i = 0; i < $max_iterations; i++) {
        body_fn(MlrtFture($tensor_list_or_flow_in[i]),
                MlrtPromise($tensor_list_or_flow_in[i+1]),
                i, i, $invariant_args);
      }

      return $tensor_list_or_flow_in[$max_iterations]
  }];

  let arguments = (ins
    TFTensorType:$max_iterations,
    Variadic<TFTensorType>:$tensor_list_or_flow_in,
    Variadic<TFTensorType>:$invariant_args,
    FlatSymbolRefAttr:$body_fn,
    I32Attr:$num_tensor_list_or_flow_in
  );

  let results = (outs
     Variadic<TFTensorType>:$result
  );

  let assemblyFormat = "`(`$max_iterations`,` $tensor_list_or_flow_in`,` $invariant_args `)` attr-dict `:` functional-type(operands, results)";
}

def AsyncWhileOp : TensorflowMlrt_Op<"async_while", [Pure]> {
  let summary = "Asynchronously execution of while op for tf_mlrt";
  let description = [{
    cond: The boolean to control whether the first iteration should be executed.
    arguments: the last $immutable_size elements are invariants between iterations.
    results: a list of futures.
    body_fn: its input are [predicate_promise, arg0_future, arg0_promise, arg1_future, arg1_promise, ..., invariant_args] and has no returned results.
  }];

  let arguments = (ins TFTensorType:$cond,
                       Variadic<TFTensorType>:$arguments,
                       FlatSymbolRefAttr:$body_fn,
                       I32Attr:$invariant_size);

  let results = (outs Variadic<MlrtFutureType>);

  let assemblyFormat = [{
    $body_fn `(` $cond `,` $arguments `)` attr-dict `:` `(` type($cond) `,` type($arguments) `)` `->` `(` type(results) `)`
  }];
}

def IfrtLoadVariableOp: TensorflowMlrt_Op<"ifrt_load_variable", [Pure]> {
  let summary = "Loads a variable tensor as an IFRT array for mlrt";

  let description = [{
    This is the MLRT version of tf.IfrtLoadVariableOp.

    This op loads a variable tensor as an IFRT array and binds it with the specified name.

    This op is an replacement of `tf.ReadVariableOp` in the case that a constant
    variable tensor is an input to the tpu program invoked by `tf.IfrtCall`.

    After a `tf.ReadVariableOp` is lowered into `tf.IfrtLoadVariableOp`, the `tf.IfrtCall` kernel
    will bind the loaded IFRT array by name with the tpu program's input.

    `tf.IfrtLoadVariableOp` converts the tensor into an IFRT array based on device and sharding
    configuration specified in `VariableDeviceShardingConfigProto`.

    This op returns a scalar string tensor as a key for user to look for the loaded array
    and a future containing the restored tensor.
  }];

  let arguments = (ins
    TFTensorType:$variable,
    StrAttr:$device_sharding_config_proto_text,
    StrAttr:$name
  );

  let results = (outs
    TFTensorType:$array_key,
    MlrtFutureType: $tensor_future
  );
}

def IfrtRestoreVariableOp: TensorflowMlrt_Op<"ifrt_restore_variable", []> {
  let summary = "Restore variable tensors";
  let description = [{
    This is the MLRT version of tf.IfrtRestoreVariableOp.

    This Op is similar to a combination of RestoreV2 and AssignVariable Op, but
    this Op's execution is asynchronous.

    This Op is specific to MLRT runtime and is not a stable interface for
    serialization.

    This Op will restore the tensors asynchronously and allow the runtime to look
    for them.
    The runtime shall handle the possibility that the tensors are not ready when requested
    because the tensors are loaded asynchronously.

  }];

  let arguments = (ins
    TFTensorType:$prefix,
    TFTensorType:$tensor_names,
    TFTensorType:$shape_and_slices,
    Variadic<TFTensorType>:$var_handles,
    TypeArrayAttr: $restored_dtypes
  );
}

#endif
