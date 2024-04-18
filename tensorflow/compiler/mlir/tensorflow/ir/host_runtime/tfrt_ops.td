/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is the operation definition file for TensorFlow operations with
// implementation available only in TFRT.

#ifndef TFRT_OPS
#define TFRT_OPS

include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "mlir/IR/OpBase.td"

def TF__TfrtSetResourceOp : TF_Op<"_TfrtSetResource", []> {
  let summary = "Set values as TFRT runtime's static resource.";

  let description = [{
    Setting a tensor value in TFRT runtime's static resource manager, using
    index as its identifier. It can be retrieved by _TfrtGetResource using the
    same index.
  }];

  let arguments = (ins
    TF_Tensor:$arg,
    I64Attr:$index
  );
}

def TF__TfrtGetResourceOp : TF_Op<"_TfrtGetResource",
    [Pure, TF_NoConstantFold, DeclareOpInterfaceMethods<TF_ResourceHandleAllocatorInterface>]> {
  let summary = "Get values from TFRT runtime's static resource.";

  let description = [{
    Getting tensor values from TFRT runtime's static resource manager, using
    the indices that are previously set in _TfrtSetResource op.
    `shared_name` and `container` are used to store identifying information for
    resource tensors that have been hoisted. This is important for resource
    alias analysis and side effect analysis. These are empty for non-resource
    type tensors.
  }];

  let arguments = (ins
    I64ArrayAttr:$indices,
    StrArrayAttr:$shared_name,
    StrArrayAttr:$container
  );

  let results = (outs
    Variadic<TF_Tensor>:$results
  );

  let hasVerifier = 1;
}

def TF_IfrtRestoreVariableOp : TF_Op<"IfrtRestoreVariableOp", []> {
  let summary = "Restore variable tensors";
  let description = [{
    This Op is similar to a combination of RestoreV2 and AssignVariable Op, but
    this Op's execution is asynchronous.

    This Op is specific to TFRT + IFRT runtime and is not a stable interface for
    serialization.

    This Op will restore the tensors asynchronously. There are N restored tensors
    corresponding to N $var_handle. Each restored tensor is given a unique name,
    typically a concatenation of `container_name` and `shared_name` from a `var_handle`.
    The runtime shall handle the possibility that the tensors are not ready when requested
    because the tensors are loaded asynchronously.
  }];

  let arguments = (ins
    Arg<TF_StrTensor, [{Must have a single element.  The prefix of a V2 checkpoint.}]>:$prefix,
    Arg<TF_StrTensor, [{shape {N}.  The names of the tensors to be restored. This is tensor name in the model.}]>:$tensor_names,
    Arg<TF_StrTensor, [{shape {N}.  The slice specs of the tensors to be restored.
Empty strings indicate that they are non-partitioned tensors.}]>:$shape_and_slices,
    // VarHandles for restored tensors.
    Variadic<TF_Tensor>:$var_handles,
    // Data types for restored tensor.
    TypeArrayAttr: $restored_dtypes
  );

}


def TF_IfrtLoadVariableOp : TF_Op<"IfrtLoadVariable", [Pure]> {
  let summary = "Loads a variable tensor as an IFRT array";

  let description = [{
    This op loads a variable tensor as an IFRT array and binds it with the specified name.

    This op is an replacement of `tf.ReadVariableOp` in the case that a constant
    variable tensor is an input to the tpu program invoked by `tf.IfrtCall`.

    After a `tf.ReadVariableOp` is lowered into `tf.IfrtLoadVariableOp`, the `tf.IfrtCall` kernel
    will bind the loaded IFRT array by name with the tpu program's input.

    `tf.IfrtLoadVariableOp` converts the tensor into an IFRT array based on device and sharding
    configuration specified in `VariableDeviceShardingConfigProto`.

    This op returns a scalar string tensor containing the loaded variable name, which can be
    used as a key to look for the loaded IFRT array in runtime and a restored tensor, which
    maybe lowered to a future by runtime.
  }];

  let arguments = (ins
    Arg<TF_Tensor, [{The variable tensor to be loaded}], []>:$variable,
    DefaultValuedStrAttr<StrAttr, "">:$device_sharding_config_proto_text,
    DefaultValuedAttr<StrAttr, "">:$name
  );

  let results = (outs
    TF_StrTensor:$array_key,
    TF_Tensor: $tensor_future
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;
}


def TF_IfrtCallOp : TF_Op<"IfrtCall", []> {
  let summary = "Invokes a program via IFRT on a device";

  let description = [{
    This op calls an IFRT program uniquely identified by the given program id.

    During lowering from a `tf_device.cluster_func` op to a `tf.IfrtCall` op,
    the region owned by the former will be outlined to a function with a
    `tfrt_ifrt_serving.program_id` attribute. After that, the runtime ensures
    that the outlined function is compiled into an executable and is available
    for lookup from `IfrtCall` TF ops.

    `variable_arg_indices` is a sorted (ascending order) array and indicates which
    element of `args` is a key to a loaded array corresponding to a variable.
  }];

  let arguments = (ins
    Variadic<TF_Tensor> : $args,
    I64Attr : $program_id,
    I32ArrayAttr : $variable_arg_indices
  );

  let results = (outs Variadic<TF_Tensor> : $results);

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;

  let hasVerifier = 1;
}

// TODO(chky): Consider adding this op to tensorflow core ops.
def TF_PwStreamResultsOp : TF_Op<"PwStreamResults"> {
  let summary = "Streams results back to the controller";

  let description = [{
    This op is a TensorFlow op that represents "streamed outputs", where
    intermediate results can be returned immediately without waiting for the
    entire signature computation to complete.

    This op takes `args` with their `names` (their cardinality must match) and
    sends the given argument tensors back to the serving controller. This
    triggers a controller-side stream callback (see `ScopedStreamCallback`).

    In addition to the listed attributes, this op has two "hidden" attributes
    that do not exist in SavedModel but are dynamically populated by the serving
    runtime:

    * `_controller_address`: Address of the remote instance to which tensors
      will be sent via e.g. RPC.
    * `_callback_id`: Identifier for the callback to be called from the
      controller. See `ScopedStreamCallback`.
  }];

  let arguments = (ins
    Variadic<TF_Tensor> : $args,
    StrArrayAttr : $names
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;

  let hasVerifier = 1;
}

#endif // TFRT_OPS
