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

#ifdef TFRT_FALLBACK_ASYNC_OPS
#else
#define TFRT_FALLBACK_ASYNC_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/compiler/opdefs/tfrt_op_interfaces.td"
include "tfrt/compiler/opdefs/tfrt_traits.td"
include "tfrt/core_runtime/opdefs/corert_base.td"
include "tfrt/core_runtime/opdefs/corert_traits.td"
include "tfrt_fallback.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def FallbackAsync_Dialect : Dialect {
  let name = "tfrt_fallback_async";

  let description = [{
    The TFRT Fallback Async Dialect.
  }];

  let cppNamespace = "::tfrt::fallback_async";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

class FallbackAsync_Op<string mnemonic, list<Trait> traits = []> :
    Op<FallbackAsync_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

def SetResourceOp : FallbackAsync_Op<"set_resource", [CoreRT_TypedAttributeTrait]> {
  let summary = "Set a tensor in resource array";

  let description = [{
    Set a tensor in resource array.

    arg: the tensor to be set in the resource array.
    index: the index in the resource array

    in_ch and out_ch are for control dependencies.
  }];

  let arguments = (ins
    TFRT_ChainType:$in_ch,
    TFTensorType:$arg,
    StrAttr:$device,
    I64Attr:$index
  );

  let results = (outs
    TFRT_ChainType:$out_ch
  );

  let assemblyFormat = "operands attr-dict";
}

def GetResourceOp : FallbackAsync_Op<"get_resource",
    [CoreRT_TypedAttributeTrait,
     TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "get a tensor in resource array";

  let description = [{
    Get a tensor in resource array.

    indices: the indices in the resource array.
    results: the tensor values for the corresponding indices.

    in_ch and out_ch are for control dependencies.
  }];

  let arguments = (ins
    TFRT_ChainType:$in_ch,
    StrAttr:$device,
    I64ArrayAttr:$indices
  );

  let results = (outs
    TFRT_ChainType:$out_ch,
    Variadic<TFTensorType>:$results
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($in_ch) `)` `->` `(` type($out_ch) `,` type($results) `)`";
}

def ConstTensorProtoOp : FallbackAsync_Op<"const_tensor_proto",
    [NoSideEffect, CoreRT_TypedAttributeTrait,
     TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "Create a tensorflow tensor from serialized tensor proto";

  let description = [{
    Create a tensorflow tensor from serilaized tensor proto.

    Example:
      %tensor = tfrt_fallback_async.const_tensor_proto "<serialized proto content>"
  }];

  let arguments = (ins
    StrAttr:$serialized_tensor_proto
  );

  let results = (outs
    TFTensorType:$result
  );

  let assemblyFormat = "$serialized_tensor_proto attr-dict";
}

def ConstDenseTensorOp : FallbackAsync_Op<"const_dense_tensor",
    [ConstantLike, NoSideEffect, CoreRT_TypedAttributeTrait,
     TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "Create a constant dense tensorflow tensor.";

  let description = [{
    ConstDenseTensorOp creates a constant dense tensorflow tensor.

    Example:
      %tensor = tfrt_fallback_async.const_dense_tensor dense<[0, 1, 2]> : tensor<3xi32>
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    TFTensorType:$result
  );

  let assemblyFormat = "$value attr-dict";

  let hasFolder = 1;
}

// TODO(b/181257202): This should be `ConstantLike`.
def ConstStringTensorOp : FallbackAsync_Op<"const_string_tensor",
    [NoSideEffect, CoreRT_TypedAttributeTrait,
     TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "Create a constant string tensorflow tensor.";
  let description = [{
    ConstStringTensorOp creates a constant string tensorflow tensor.

    Example:
      %tensor = tfrt_fallback_async.const_string_tensor {shape = [1, 2], value = ["const", "string"]}

    Note: if the entries in `value` are all the same, we can just keep one copy.
      In other words, the following two lines are effectively the same.
      %tensor = tfrt_fallback_async.const_string_tensor {shape = [1, 2], value = ["string", "string"]}
      %tensor = tfrt_fallback_async.const_string_tensor {shape = [1, 2], value = ["string"]}
  }];

  let arguments = (ins
    I64ArrayAttr:$shape,
    StrArrayAttr:$value
  );

  let results = (outs
    TFTensorType:$result
  );

  let assemblyFormat = "attr-dict";
}

def FallbackTensorToCoreRTTensorHandleOp : FallbackAsync_Op<
                              "fallback_tensor_to_corert_tensorhandle",
                              [NoSideEffect, CoreRT_TypedAttributeTrait,
                               TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "Convert fallback tensor to corert tensorhandle";

  let description = [{
    Convert the fallback tensor to corert tensorhandle.

    operands: a list of fallback tensors.
    device: the name of the tensorflow device that is associated with the
      input fallback tensors.

    results: the converted corert tensorhandles.
  }];

  let arguments = (ins
    Variadic<TFTensorType>:$operands,
    StrAttr:$device
  );

  let results = (outs
    Variadic<CoreRT_TensorHandleType>:$results
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($operands) `)` `->` `(` type($results) `)`";
}

def CoreRTTensorHandleToFallbackTensorOp : FallbackAsync_Op<
                              "corert_tensorhandle_to_fallback_tensor",
                              [NoSideEffect, CoreRT_TypedAttributeTrait,
                               TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "Convert corert tensorhandle to fallback tensor";

  let description = [{
    Convert the corert tensorhandle to the fallback tensor.

    operands: a list of corert tensorhandles.
    device: the name of the tensorflow device that is associated with the
      output fallback tensor.

    results: the converted fallback tensors.
  }];

  let arguments = (ins
    Variadic<CoreRT_TensorHandleType>:$operands,
    StrAttr:$device
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($operands) `)` `->` `(` type($results) `)`";

  let hasCanonicalizer = 1;
}

def PredicateOp : FallbackAsync_Op<"predicate",
    [NoSideEffect, CoreRT_TypedAttributeTrait,
     TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
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

def CopyIfSmallOp : FallbackAsync_Op<"copy_if_small",
    [NoSideEffect, CoreRT_TypedAttributeTrait,
     TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "Copy the CPU fallback tensor if it is small";

  let description = [{
    This kernel performs deep copy on the input tensor if it is small, to avoid
    atomic contention on its refcount.

    Note that this kernel always create a new AsyncValue for each result to
    avoid atomic contention on AsyncValue's refcount.
  }];

  let arguments = (ins
    TFTensorType:$operand
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let assemblyFormat = "$operand attr-dict `:` `(` type($operand) `)` `->` `(` type($results) `)`";
}

def CreateOp: FallbackAsync_Op<"createop", [CoreRT_TypedAttributeTrait]> {
  let summary = "The Fallback CreateOp";

  let description = [{
    The CreateOp creates the tensorflow::OpKernel in the fallback context.

    op_key: The unique key to identify this op.
    num_args: The number of inputs.
    device: The tensorflow device. eg. "/CPU:0"
    op_attrs: The tensorflow attributes excluding the func attrs.
    op_func_attrs: The func attrs.
    op_name: The tensorflow op name. eg. "tf.AddV2"

    Example:
      %out_ch = tfrt_fallback_async.createop(%in_ch) key(0) device("/CPU:0")
        "some.op"() {attr1 = value, attr2 = value2} {f_attr1 = "fn1", f_attr2 = "fn2"} num_args(1)
  }];

  let arguments = (ins
    TFRT_ChainType:$in_ch,
    I64Attr:$num_args,
    StrAttr:$device,
    ArrayAttr:$op_attrs,
    // TODO(b/173025975): consider using DictionaryAttr after we support
    // BEF conversion for this type.
    ArrayAttr:$op_func_attrs,
    I64Attr:$op_key,
    StrAttr:$op_name
  );

  let results = (outs
    TFRT_ChainType:$out_ch
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def ExecuteOp : FallbackAsync_Op<"executeop",
    [NoSideEffect, CoreRT_TypedAttributeTrait,
     TFRT_CostFunctionInterface, TFRT_AttrCostTrait]> {
  let summary = "The Fallback ExecuteOp";
  let description = [{
    The ExecuteOp executes an operation on the specified device.

    Example:
      %res = tfrt_fallback_async.executeop key(0) cost(100) device("/CPU:0") "some.op"(%arg) : 1

    Note that the trailing number indicates the number of results.

    The func attrs and op attrs are placed in two separate regions.

    Example:
      %res = tfrt_fallback_async.executeop key(0) device("/CPU:0") "some.op"(%arg)
          {attr1 = value, attr2 = value2} {f_attr1 = "fn1", f_attr2 = "fn2"} : 1

    Note that in this case, the first bracket contains op attributes other than
    function attributes, and the second braket contains function attributes,
    which represent function names.

    Refer to b/172597167 for approaches considered for supporting function
    attributes.
  }];

  let arguments = (ins
    Variadic<TFTensorType>:$operands,
    StrAttr:$device,
    ArrayAttr:$op_attrs,
    // TODO(b/173025975): consider using DictionaryAttr after we support
    // BEF conversion for this type.
    ArrayAttr:$op_func_attrs,
    I64Attr:$op_key,
    StrAttr:$op_name,
    I64Attr:$_tfrt_cost
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let extraClassDeclaration = [{
    void getOpAttrs(SmallVectorImpl<std::pair<::llvm::StringRef, ::mlir::Attribute>>* op_attrs);
  }];

  let builders = [
    OpBuilder<(ins "ArrayRef<Type>":$results, "ValueRange":$operands,
      "StringRef":$device,
      "::llvm::ArrayRef<std::pair<::llvm::StringRef, ::mlir::Attribute>>":$op_attrs,
      "::llvm::ArrayRef<std::pair<::llvm::StringRef, ::mlir::Attribute>>":$op_func_attrs,
      "int64_t":$op_key, "::llvm::StringRef":$op_name, "int64_t":$_tfrt_cost)>];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def ExecuteOpSeq : FallbackAsync_Op<"executeop.seq",
    [CoreRT_TypedAttributeTrait,
     TFRT_CostFunctionInterface, TFRT_AttrCostTrait]> {
  let summary = "The sequenced version of Fallback ExecuteOp";
  let description = [{
    The ExecuteOpSeq executes an operation on the specified device, similar to
    ExecuteOp. In addition, it takes an in_op_chain and produces an out_op_chain
    that can be used to sequence op executions. ExecuteOpSeq is nonstrict on
    in_op_chain.

    Example:
      %op_ch_out, %res = tfrt_fallback_async.executeop.seq(%op_ch_in)
        key(0) cost(100) device("/CPU:0") "some.op"(%arg) : 1

    Note that the trailing number indicates the number of results.

    The func attrs and op attrs are placed in two separate regions.

    Example:
      %op_ch_out, %res = tfrt_fallback_async.executeop.seq(%op_ch_in)
        key(0) cost(100) device("/CPU:0") "some.op"(%arg)
        {fn = function_1} {attr1 = value, attr2 = value2} : 1

    Note that in this case, the first braket contains op attributes other than
    function attributes, and the second braket contains function attributes,
    which represent function names.

    Refer to b/172597167 for approaches considered for supporting function
    attributes.
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    Variadic<TFTensorType>:$operands,
    StrAttr:$device,
    ArrayAttr:$op_attrs,
    // TODO(b/173025975): consider using DictionaryAttr after we support
    // BEF conversion for this type.
    ArrayAttr:$op_func_attrs,
    I64Attr:$op_key,
    StrAttr:$op_name,
    I64Attr:$_tfrt_cost
  );

  let results = (outs
    TFRT_ChainType:$out_op_chain,
    Variadic<TFTensorType>:$results
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def BatchFunctionOp : FallbackAsync_Op<"batch_function", [CoreRT_TypedAttributeTrait]> {
  let summary = "Fallback ExecuteOp specialized for tf.BatchFunction.";
  let description = [{
    This kernel executes a variant tf.BatchFunction kernel that supports having
    the `f` attribute as a BEF function.

    Example:
      %op_ch_out, %res = tfrt_fallback_async.atch_function(%op_ch_in) @f (%input, %captured_input) {
          num_batch_threads = 1,
          max_batch_size = 4,
          allowed_batch_sizes = [2 , 4],
          batch_timeout_micros = 50,
          container = "container",
          shared_name = "shared_name",
          batching_queue = "batching_queue",
          enable_large_batch_splitting = false,
          Tin = [f32],
          Tcaptured = [f32],
          Tout = [f32]} : 1

    Note that the trailing number indicates the number of results.
  }];

  let arguments = (ins
                   TFRT_ChainType:$in_op_chain,
    Variadic<CoreRT_TensorHandleType>:$operands,
    SymbolRefAttr:$f,
    ArrayAttr:$op_attrs
  );

  let results = (outs
    TFRT_ChainType:$out_op_chain,
    Variadic<CoreRT_TensorHandleType>:$results
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def ExecuteOpWithAllocator : FallbackAsync_Op<"executeop.allocator",
    [NoSideEffect, CoreRT_TypedAttributeTrait, TFRT_CostFunctionInterface, TFRT_AttrCostTrait]> {
  let summary = "The Fallback ExecuteOp with custom allocator";
  let description = [{
    Similar to ExecuteOp but takes a custom allocator for allocating output tensors.
  }];

  let arguments = (ins
    TFAllocatorType:$allocator,
    Variadic<TFTensorType>:$operands,
    StrAttr:$device,
    ArrayAttr:$op_attrs,
    ArrayAttr:$op_func_attrs,
    I64Attr:$op_key,
    StrAttr:$op_name,
    I64Attr:$_tfrt_cost
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def ExecuteOpSeqWithAllocator : FallbackAsync_Op<"executeop.seq.allocator",
    [CoreRT_TypedAttributeTrait, TFRT_CostFunctionInterface, TFRT_AttrCostTrait]> {
  let summary = "The sequenced version of Fallback ExecuteOp with custom allocator";
  let description = [{
    Similar to ExecuteOpSeq but takes a custom allocator for allocating output tensors.

    Example:
      %op_ch_out, %res = tfrt_fallback_async.executeop.seq.allocator(%op_ch_in, %allocator)
        key(0) cost(100) device("/CPU:0") "some.op"(%arg) : 1
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    TFAllocatorType:$allocator,
    Variadic<TFTensorType>:$operands,
    StrAttr:$device,
    ArrayAttr:$op_attrs,
    // TODO(b/173025975): consider using DictionaryAttr after we support
    // BEF conversion for this type.
    ArrayAttr:$op_func_attrs,
    I64Attr:$op_key,
    StrAttr:$op_name,
    I64Attr:$_tfrt_cost
  );

  let results = (outs
    TFRT_ChainType:$out_op_chain,
    Variadic<TFTensorType>:$results
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif
