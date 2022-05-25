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

#ifdef TFRT_FALLBACK_SYNC_OPS
#else
#define TFRT_FALLBACK_SYNC_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/core_runtime/opdefs/corert_base.td"
include "tfrt/core_runtime/opdefs/corert_traits.td"
include "tfrt_fallback.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def FallbackSync_Dialect : Dialect {
  let name = "tfrt_fallback_sync";

  let description = [{
    The TFRT Fallback Sync Dialect.
  }];

  let cppNamespace = "::tfrt::fallback_sync";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

class FallbackSync_Op<string mnemonic, list<Trait> traits = []> :
    Op<FallbackSync_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

// TODO(rohitju): Add a verifier if there is anything that should be verified
// compile time.
def SetResourceOp : FallbackSync_Op<"set_resource", [CoreRT_TypedAttributeTrait]> {
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

def GetResourceOp : FallbackSync_Op<"get_resource",
    [CoreRT_TypedAttributeTrait]> {
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

def CreateOp: FallbackSync_Op<"createop", [CoreRT_TypedAttributeTrait]> {
  let summary = "The Fallback CreateOp";

  let description = [{
    The CreateOp creates the tensorflow::OpKernel in the fallback context.

    op_key: The unique key to identify this op.
    num_args: The number of inputs.
    device: The tensorflow device. eg. "/CPU:0"
    op_attrs: The tensorflow attributes excluding the func attrs.
    op_name: The tensorflow op name. eg. "tf.AddV2"

    Example:
      tfrt_fallback_async.createop() key(0) device("/CPU:0")
        "some.op"() {attr1 = value, attr2 = value2} {f_attr1 = "fn1", f_attr2 = "fn2"} num_args(1)
  }];

  let arguments = (ins
    I64Attr:$num_args,
    ArrayAttr:$op_attrs,
    I64Attr:$op_key,
    StrAttr:$op_name
  );

  let results = (outs);
}

def ExecuteOp : FallbackSync_Op<"executeop",
    [NoSideEffect, CoreRT_TypedAttributeTrait]> {
  let summary = "The Fallback Sync ExecuteOp";
  let description = [{
    The ExecuteOp executes an operation on the specified device.

    Example:
      %res = tfrt_fallback_sync.executeop "some.op"(%arg) : 1

    Note that the trailing number indicates the number of results.
  }];

  let arguments = (ins
    Variadic<TensorType>:$operands,
    ArrayAttr:$op_attrs,
    StrAttr:$op_name
  );

  let results = (outs
    Variadic<TensorType>:$results
  );

  let extraClassDeclaration = [{
    void getOpAttrs(SmallVectorImpl<std::pair<StringRef, Attribute>>* op_attrs);
  }];

  let builders = [
    OpBuilder<(ins "ArrayRef<Type>":$results, "ValueRange":$operands,
      "ArrayRef<std::pair<StringRef, Attribute>>":$op_attrs,
      "StringRef":$op_name)>];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif
