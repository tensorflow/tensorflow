/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#ifdef MLRT_TF_OPS
#else
#define MLRT_TF_OPS

include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_dialect.td"
include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "tfrt/compiler/opdefs/tfrt_op_interfaces.td"  // from @tf_runtime
include "tfrt/compiler/opdefs/tfrt_traits.td"  // from @tf_runtime

// tf_mlrt.tf_await returns a tensorflow Tensor. It is a fake op that is only
// used during parallelization and has no runtime implementation.
def TFAwaitOp: TensorflowMlrt_Op<"tf_await", [Pure, TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
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
    TF_Tensor:$result
  );
}

// tf_mlrt.tf_promise takes a tensorflow Tensor. It is a fake op that is only
// used during parallelization and has no runtime implementation.
def TFPromiseOp: TensorflowMlrt_Op<"tf_promise", [TF_MustExecute, TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "Set a tensor in a promise";

  let description = [{
    Set a tensor in a promise.

    $promise: A value of type !mlrt.promise. The underlying value will always be a tensorflow tensor.
    $tensor: A tensorflow tensor.
  }];

  let arguments = (ins
    MlrtPromiseType:$promise,
    TF_Tensor:$tensor
  );
}

def TFMapFnOp : TensorflowMlrt_Op<"tf_map_fn", [AttrSizedOperandSegments, Pure]> {
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
    TF_Tensor:$max_iterations,
    Variadic<TF_Tensor>:$tensor_list_or_flow_in,
    Variadic<TF_Tensor>:$invariant_args,
    FlatSymbolRefAttr:$body_fn,
    I32Attr:$num_tensor_list_or_flow_in
  );

  let results = (outs
     Variadic<TF_Tensor>:$result
  );

  let assemblyFormat = "`(`$max_iterations`,` $tensor_list_or_flow_in`,` $invariant_args `)` attr-dict `:` functional-type(operands, results)";
}

def TFAsyncWhileOp : TensorflowMlrt_Op<"tf_async_while", [Pure]> {
  let summary = "Asynchronously execution of while op for tf_mlrt";
  let description = [{
    cond: The boolean to control whether the first iteration should be executed.
    arguments: The last $invariant_size elements are invariants between iterations.
    results: a list of futures.
    body_fn: input is [predicate_promise, arg0_future, arg0_promise, arg1_future, arg1_promise, ..., invariant_args] and has no returned results.
  }];

  let arguments = (ins TF_Tensor:$cond,
                       Variadic<TF_Tensor>:$arguments,
                       FlatSymbolRefAttr:$body_fn,
                       I32Attr:$invariant_size);

  let results = (outs Variadic<MlrtFutureType>);

  let assemblyFormat = [{
    $body_fn `(` $cond `,` $arguments `)` attr-dict `:` `(` type($cond) `,` type($arguments) `)` `->` `(` type(results) `)`
  }];
}

def TFTPUCompileAndExecuteOp : TensorflowMlrt_Op<"tf_tpu_compile_and_execute", [TF_MustExecute]> {
  let summary = "tpu compile and execute operation";
  let description = [{
    tf_mlrt_tpu.compile_and_execute compiles a mlir tpu program and executes the compiled tpu program.

    $mlir_module is a serialized MLIR module with a `main` function that contains target computation.
    $metadata is a serialized TPUCompileMetadataProto describing the shapes and types of the inputs to the computation, as well as a mapping onto the TPU pod topology.
    $constant_operand_indices are the indices of the inputs that are constant to the TPU program (e.g. weights in inference), the rest of the inputs are input tensors.
    constant_operand_indices is sorted in ascending order.
    $operands_with_static_shape are indices of operands that are tagged with a maximum static shape.
    $producer_name is a string describing the name of the framework that added support for running this portion of the model on TPUs.

    Example:
      %rendezvous_key_base, %result = tf_mlrt_tpu.compile_and_execute (%operands) constant_operand_indices = [1, 3] metadata = "metadata..." mlir_module = "mlir_module..."
  }];
  let arguments = (ins
    Variadic<TF_Tensor>:$operands_and_static_shapes,
    DenseI32ArrayAttr:$constant_operand_indices,
    StrAttr:$metadata,
    StrAttr:$mlir_module,
    UI32Attr:$num_operands,
    DenseI32ArrayAttr:$operands_with_static_shape,
    StrAttr:$producer_name
  );

  let results = (outs
    TF_Tensor:$rendezvous_key_base,
    Variadic<TF_Tensor>:$results
  );

  let assemblyFormat = [{
    `(` $operands_and_static_shapes `)` attr-dict `:` functional-type($operands_and_static_shapes, results)
  }];
}

def TFIfrtLoadVariableOp: TensorflowMlrt_Op<"tf_ifrt_load_variable", [Pure]> {
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
    TF_Tensor:$variable,
    StrAttr:$device_sharding_config_proto_text,
    StrAttr:$name
  );

  let results = (outs
    TF_Tensor:$array_key,
    MlrtFutureType: $tensor_future
  );
}



#endif
