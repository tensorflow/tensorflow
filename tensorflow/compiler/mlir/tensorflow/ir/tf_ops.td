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

// This is the operation definition file for TensorFlow.
//
// This file contains TensorFlow ops whose definitions are amended to fix
// issues or provide more information. In this file you have full control
// of the op definition; all changes will be retained with subsequent
// refreshes.
//
// This file includes another file, `tf_generated_ops.td`, which contains
// all ops whose definitions are generated from TensorFlow codebase.
// Changes made there are not respected.

#ifndef TF_OPS
#define TF_OPS

include "tensorflow/compiler/mlir/tensorflow/ir/tf_generated_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

class TF_TensorListInitOp<string mnemonic> : TF_Op<mnemonic, [NoSideEffect]> {
  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeAttr shape_type = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    // This is required to populate derived attributes during export in a
    // meaningful way. Else during export to GraphDef element_type() query
    // will result in out of bounds access/assert.
    if (handle_dtype().getSubtypes().size() != 1) {
      return emitOpError(
          "must have exactly one subtype in the result variant type");
    }

    return Verify(*this);
  }];

  DerivedTypeAttr element_dtype = DerivedTypeAttr<
      "return getElementTypeOrSelf(element_type());">;

  let extraClassDeclaration = [{
    // Returns type of the TensorList element produced by this op.
    TensorType element_type() { return handle_dtype().getSubtypes()[0]; }

    // Returns data type of the result handle. Returned type contains type of
    // the TensorList element as a subtype.
    VariantType handle_dtype() {
      return getElementTypeOrSelf(handle().getType()).cast<TF::VariantType>();
    }
  }];
}

// In MLIR, the TensorFlow tensor value is represented as an ElementsAttr, with
// its type encoding the tensor's shape and data type.
def TF_ConstOp : TF_Op<"Const", [ConstantLike, NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constant tensor op";

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let builders = [
    OpBuilder<
      "OpBuilder &builder, OperationState &result, Attribute value">,
    OpBuilder<
      "OpBuilder &builder, OperationState &result, Type type, Attribute value">,
  ];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(ArrayRef<Type> l, ArrayRef<Type> r) {
      return BroadcastCompatible(l, r);
    }
  }];
}

def TF_DataFormatVecPermuteOp : TF_Op<"DataFormatVecPermute", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Permute input tensor from `src_format` to `dst_format`";

  let description = [{
Input tensor must be a vector of size 4, or a 4x2 tensor.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$x,

    DefaultValuedAttr<StrAttr, "NHWC">:$src_format,
    DefaultValuedAttr<StrAttr, "NCHW">:$dst_format
  );

  let results = (outs
    TF_I32OrI64Tensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{ return Verify(*this); }];
}

def TF_EmptyTensorListOp : TF_TensorListInitOp<"EmptyTensorList"> {
  let summary = "Creates and returns an empty tensor list.";

  let description = [{
All list elements must be tensors of dtype element_dtype and shape compatible
with element_shape.

handle: an empty tensor list.
element_dtype: the type of elements in the list.
element_shape: a shape compatible with that of elements in the list.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$element_shape,
    I32Tensor:$max_num_elements
  );
}

// TODO(fengliuai): The tf.Identity is side-effect free and it doesn't change
// the status of the system during the execution. However it shouldn't be folded
// in general if it used to serve for caching and some other invariant checks,
// so we removed the side-effect free property in the op definition. This is a
// hack, and we should fix it if we have a better way to model it.
def TF_IdentityOp : TF_Op<"Identity", [TF_OperandsSameAsResultsTypeOrRef]> {
  let summary = "Identity op";

  let description = [{
Returns a tensor with the same shape and contents as input.
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr T = TF_DerivedResultTypeAttr<0>;
}

def TF_IfOp : TF_Op<"If", []> {
  let summary = "output = cond ? then_branch(input) : else_branch(input)";

  let description = [{
output = cond ? then_branch(input) : else_branch(input)

cond: A Tensor. If the tensor is a scalar of non-boolean type, the
    scalar is converted to a boolean according to the
    following rule: if the scalar is a numerical value, non-zero means
    True and zero means False; if the scalar is a string, non-empty
    means True and empty means False. If the tensor is not a scalar,
    being empty means False and being non-empty means True.
input: A list of input tensors.
then_branch: A function that takes 'inputs' and returns a list of
    tensors, whose types are the same as what else_branch returns.
else_branch: A function that takes 'inputs' and returns a list of
    tensors.  whose types are the same as what then_branch returns.
  }];

  let arguments = (ins
    TF_Tensor:$cond,
    Variadic<TF_Tensor>:$input,

    FlatSymbolRefAttr:$then_branch,
    FlatSymbolRefAttr:$else_branch,
    DefaultValuedAttr<TF_ShapeAttrArray, "{}">:$output_shapes,

    // Used to map StatelessIf and If op defined in TensorFlow to a common op.
    BoolAttr:$is_stateless
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeAttr Tcond = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<1>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_MeanOp : TF_Op<"Mean", [NoSideEffect, TF_FoldOperandsTransposeInterface]> {
  let summary = "Computes the mean of elements across dimensions of a tensor.";

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    TF_NumberTensor:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TF_NumberTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;

  let extraClassDeclaration = [{
    // TF_FoldOperandsTransposeInterface:
    SmallVector<unsigned, 4> GetLayoutDependentArgs() { return {0}; }
    SmallVector<unsigned, 4> GetLayoutDependentResults() { return {}; }
    LogicalResult FoldOperandsPermutation(ArrayRef<int64_t> permutation);
  }];
}

def TF_LegacyCallOp : TF_Op<"LegacyCall",
                            [CallOpInterface, NoSideEffect]> {
  let summary =
    "returns `f(inputs)`, where `f` is a function.";

  let description = [{
    The LegacyCall operation represents a direct call to a function that is
    within the same symbol scope as the call and is mapped to a GraphDef node
    with the function name as the op name. Unlike a PartitionedCall which
    represents asynchronously executing a function across multiple devices, a
    LegacyCall represents a function call with the only attribute
    _diable_call_shape_inference.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$args,

    FlatSymbolRefAttr:$f,
    DefaultValuedAttr<BoolAttr, "false">:$_disable_call_shape_inference
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return args(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("f");
    }
  }];
}

def TF_ParseExampleV2Op : TF_Op<"ParseExampleV2",
                                [NoSideEffect,
                                 AttrSizedResultSegments]> {

  let summary =
    "Transforms a vector of tf.Example protos (as strings) into typed tensors.";

  let arguments = (ins
    TF_StrTensor:$serialized,
    TF_StrTensor:$names,
    TF_StrTensor:$sparse_keys,
    TF_StrTensor:$dense_keys,
    TF_StrTensor:$ragged_keys,
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$dense_defaults,

    Confined<I64Attr, [IntMinValue<0>]>:$num_sparse,
    TF_ShapeAttrArray:$dense_shapes,
    I32ElementsAttr:$result_segment_sizes
  );

  let results = (outs
    Variadic<I64Tensor>:$sparse_indices,                    // len(sparse_types)
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$sparse_values,  // len(sparse_types)
    Variadic<I64Tensor>:$sparse_shapes,                     // len(sparse_types)
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$dense_values,   // len(Tdense)
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$ragged_values,  // len(ragged_value_types)
                                                            //     = len(ragged_split_types)
    Variadic<TensorOf<[I32, I64]>>:$ragged_row_splits       // len(ragged_split_types)
                                                            //     = len(ragged_value_types)
  );

  // The Verify(ParseExampleV2Op) function validates that the lengths and types
  // of these attrs are compatible.
  TF_DerivedOperandTypeListAttr Tdense = TF_DerivedOperandTypeListAttr<5>;
  TF_DerivedResultTypeListAttr sparse_types = TF_DerivedResultTypeListAttr<1>;
  TF_DerivedResultTypeListAttr ragged_value_types =
    TF_DerivedResultTypeListAttr<4>;
  TF_DerivedResultTypeListAttr ragged_split_types =
    TF_DerivedResultTypeListAttr<5>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_PartitionedCallOp : TF_Op<"PartitionedCall",
                                 [CallOpInterface, NoSideEffect]> {
  let summary =
    "returns `f(inputs)`, where `f`'s body is placed and partitioned.";

  let description = [{
Asynchronously executes a function, potentially across multiple devices but
within a single process. The kernel places and partitions a given function's
underlying graph, and executes each of the partitioned subgraphs as a function.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$args,

    SymbolRefAttr:$f,
    StrAttr:$config,
    StrAttr:$config_proto,
    StrAttr:$executor_type
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return args(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("f");
    }
  }];

  let verifier = [{ return VerifyPartitionedCall(*this); }];
}

def TF_PlaceholderOp : TF_Op<"Placeholder", [NoSideEffect]> {
  let summary = "Placeholder op";

  let description = [{
Inserts a placeholder for a tensor that will be always fed.
  }];

  let arguments = (ins
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_PlaceholderWithDefaultOp : TF_Op<"PlaceholderWithDefault", [NoSideEffect]> {
  let summary = "Placeholder op";
  let description = [{
    A placeholder op that passes through input when its output is not fed.
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
  DerivedAttr shape = TF_DerivedResultShapeAttr;
}

def TF_SparseMatMulOp : TF_Op<"SparseMatMul", [NoSideEffect]> {
  let summary = [{
SparseMatMul is MatMul with hints on the sparseness of the matrices.
  }];

  let description = [{
Similar to MatMul, with a_is_sparse and b_is_sparse indicating whether a and b
are sparse matrices.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$a,
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$b,

    DefaultValuedAttr<BoolAttr, "true">:$a_is_sparse,
    DefaultValuedAttr<BoolAttr, "false">:$b_is_sparse,

    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I32, I64, TF_Complex128, TF_Complex64]>:$product
  );

  TF_DerivedOperandTypeAttr Ta = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tb = TF_DerivedOperandTypeAttr<1>;
}


def TF_StatefulPartitionedCallOp : TF_Op<"StatefulPartitionedCall",
                                         [CallOpInterface]> {
  let summary =
    "returns `f(inputs)`, where `f`'s body is placed and partitioned.";

  let description = [{
Asynchronously executes a function, potentially across multiple devices but
within a single process. The kernel places and partitions a given function's
underlying graph, and executes each of the partitioned subgraphs as a function.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$args,

    FlatSymbolRefAttr:$f,
    StrAttr:$config,
    StrAttr:$config_proto,
    StrAttr:$executor_type
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return args(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("f");
    }
  }];

  let verifier = [{ return VerifyPartitionedCall(*this); }];
}

def TF_WhileOp : TF_Op<"While", []> {
  let summary = [{
output = input; While (Cond(output)) { output = Body(output) }
  }];

  let description = [{
output = input; While (Cond(output)) { output = Body(output) }

input: A list of input tensors whose types are T.
output: A list of output tensors whose types are T.
cond: A function takes 'input' and returns a tensor.  If the tensor is
    a scalar of non-boolean, the scalar is converted to a boolean
    according to the following rule: if the scalar is a numerical
    value, non-zero means True and zero means False; if the scalar is
    a string, non-empty means True and empty means False. If the
    tensor is not a scalar, non-emptiness means True and False
    otherwise.
body: A function that takes a list of tensors and returns another
      list of tensors. Both lists have the same types as specified
      by T.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$input,

    FlatSymbolRefAttr:$cond,
    FlatSymbolRefAttr:$body,
    DefaultValuedAttr<TF_ShapeAttrArray, "{}">:$output_shapes,
    DefaultValuedAttr<I64Attr, "10">:$parallel_iterations,

    // Used to map StatelessWhile and While op defined in TensorFlow to a common
    // op.
    BoolAttr:$is_stateless
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_TensorListReserveOp : TF_TensorListInitOp<"TensorListReserve"> {
  let summary = "List of the given size with empty elements.";

  let description = [{
element_shape: the shape of the future elements of the list
num_elements: the number of elements to reserve
handle: the output list
element_dtype: the desired type of elements in the list.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$element_shape,
    I32Tensor:$num_elements
  );
}

// This operation when auto-generated is marked as NoSideEffect because it isn't
// stateful in TensorFlow. However it is kept alive through control dependency,
// and does not have any output. When placed in an island it wouldn't be kept
// alive in any way and the canonicalizer would just always fold it away.
def TF_TPUReplicateMetadataOp : TF_Op<"TPUReplicateMetadata", []> {
  let summary = [{
Metadata indicating how the TPU computation should be replicated.
  }];

  let description = [{
This operation holds the metadata common to operations of a `tpu.replicate()` computation subgraph.
  }];

  let arguments = (ins
    Confined<I64Attr, [IntMinValue<0>]>:$num_replicas,
    DefaultValuedAttr<I64Attr, "1">:$num_cores_per_replica,
    StrAttr:$topology,
    DefaultValuedAttr<BoolAttr, "true">:$use_tpu,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$device_assignment,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$computation_shape,
    DefaultValuedAttr<StrArrayAttr, "{}">:$host_compute_core,
    DefaultValuedAttr<StrArrayAttr, "{}">:$padding_map,
    DefaultValuedAttr<StrAttr, "STEP_MARK_AT_ENTRY">:$step_marker_location,
    DefaultValuedAttr<BoolAttr, "false">:$allow_soft_placement
  );

  let results = (outs);
}

def TF_VarHandleOp : TF_Op<"VarHandleOp", []> {
  let summary = "Creates a handle to a Variable resource from its name.";

  let description = [{
container: the container this variable is placed in.
shared_name: the name by which this variable is referred to.
dtype and shape: attributes representing the data type and shape held in the
  variable.

Example:
    resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[8, 16], container="foo", shared_name="bar")
  returns a handle for a variable with name "bar" in container "foo", and the
  variable holds a tensor of shape [8, 16] and dtype int32.
  }];

  let arguments = (ins
    DefaultValuedAttr<StrAttr, "">:$container,
    DefaultValuedAttr<StrAttr, "">:$shared_name
  );

  let results = (outs
    TF_ResourceTensor:$resource
  );

  TF_DerivedOperandOrResultHandleTypeAttr dtype =
    TF_DerivedOperandOrResultHandleTypeAttr<"resource">;
  TF_DerivedOperandOrResultHandleShapeAttr shape =
    TF_DerivedOperandOrResultHandleShapeAttr<"resource">;
}

// Not generated because it begins with an underscore, which isn't allowed by
// the C++ standard.
def TF_FusedBatchNormExOp : TF_Op<"_FusedBatchNormEx", [NoSideEffect]> {
  let summary = "Internal FusedBatchNorm operation: reserved for internal use";

  let description = [{
 Do not invoke this operator directly in Python. A fusion optimization is
 expected to create these operators.
  }];

  let arguments = (ins
    TensorOf<[F16, F32]>:$x,
    F32Tensor:$scale,
    F32Tensor:$offset,
    F32Tensor:$mean,
    F32Tensor:$variance,
    Variadic<TensorOf<[F16, F32]>>:$side_input,

    DefaultValuedAttr<F32Attr, "0.0001f">:$epsilon,
    DefaultValuedAttr<F32Attr, "1.0f">:$exponential_avg_factor,
    DefaultValuedAttr<StrAttr, "Identity">:$activation_mode,
    DefaultValuedAttr<TF_ConvnetDataFormatAttr, "NHWC">:$data_format,
    DefaultValuedAttr<BoolAttr, "true">:$is_training
  );

  let results = (outs
    TensorOf<[F16, F32]>:$y,
    F32Tensor:$batch_mean,
    F32Tensor:$batch_variance,
    F32Tensor:$reserve_space_1,
    F32Tensor:$reserve_space_2,
    F32Tensor:$reserve_space_3
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr U = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedOperandSizeAttr num_side_inputs = TF_DerivedOperandSizeAttr<5>;
}

// Multiple variadic operands with different sizes are not supported by the
// dialect generator, so we manually added the op.
def TF_SendTPUEmbeddingGradientsOp : TF_Op<"SendTPUEmbeddingGradients", [AttrSizedOperandSegments]> {
  let summary = "Performs gradient updates of embedding tables.";

  let description = [{
inputs: A TensorList of gradients with which to update embedding tables.
    This argument has the same length and shapes as the return value of
    RecvTPUEmbeddingActivations, but contains gradients of the model's loss
    with respect to the embedding activations. The embedding tables are updated
    from these gradients via the optimizer specified in the TPU embedding
    configuration given to tpu.initialize_system.
learning_rates: A TensorList of float32 scalars, one for each dynamic learning
    rate tag: see the comments in
    //third_party/tensorflow/core/protobuf/tpu/optimization_parameters.proto.
    Multiple tables can share the same dynamic learning rate tag as specified
    in the configuration. If the learning rates for all tables are constant,
    this list should be empty.
config: Serialized TPUEmbeddingConfiguration proto.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,
    Variadic<TF_Tensor>:$learning_rates,
    StrAttr:$config
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandSizeAttr NN = TF_DerivedOperandSizeAttr<1>;
}

// Multiple variadic operands with different sizes are not supported by the
// dialect generator, so we manually added the op.
def TF__SendTPUEmbeddingGradientsOp : TF_Op<"_SendTPUEmbeddingGradients", [AttrSizedOperandSegments]> {
  let summary = "Performs gradient updates of embedding tables.";

  let description = [{
The gradients argument is a TensorList having the same length and shapes as the
return value of _RecvTPUEmbeddingActivations, but contains gradients of the
model's loss with respect to the embedding activations. The embedding tables are
updated from these gradients via the optimizer specified in the
TPUEmbeddingConfiguration proto given to tpu.initialize_system.

gradients: A TensorList of gradients with which to update embedding tables.
learning_rates: A TensorList of learning rates used for updating the embedding
    tables via the optimizer. The length of the TensorList must be equal to the
    number of dynamic learning rate tags specified in the
    TPUEmbeddingConfiguration proto.
deduplication_data: A Tensor with type=DT_VARIANT containing the deduplication
    data. The tensor is an XLA nested tuple containing N elements. Each
    element of the nested tuple is a tuple of rank 1 tensors. Each tensor either
    contains indices (DT_INT32) for embedding lookup or weights (DT_FLOAT) to
    apply to the output of the embedding lookup operation.
config: Serialized TPUEmbeddingConfiguration proto.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$gradients,
    Variadic<TF_Tensor>:$learning_rates,
    TF_VariantTensor:$deduplication_data,
    StrAttr:$config
  );

  TF_DerivedOperandSizeAttr NumTables = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandSizeAttr NumLearningRateTags = TF_DerivedOperandSizeAttr<1>;
}

// Updated the op description text from the auto-generated op definition.
def TF__RecvTPUEmbeddingDeduplicationDataOp : TF_Op<"_RecvTPUEmbeddingDeduplicationData", []> {
  let summary = [{
Receives deduplication data (indices and weights).
  }];

  let description = [{
The deduplication data is a Tensor with type=DT_VARIANT. The tensor itself is an
XLA nested tuple containing N elements. Each element of the nested tuple is a
tuple of rank 1 tensors. Each tensor either contains indices (DT_INT32) for
embedding lookup or weights (DT_FLOAT) to apply to the output of the embedding
lookup operation.
  }];

  let arguments = (ins
    StrAttr:$config
  );

  let results = (outs
    TF_VariantTensor:$output
  );
}

def TF_XlaShardingOp : TF_Op<"XlaSharding", [NoSideEffect]> {
  let summary = [{
An op which shards the input based on the given sharding attribute.
  }];

  let description = [{
  }];

  let arguments = (ins
    TF_Tensor:$input,

    OptionalAttr<StrAttr>:$_XlaSharding
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_InfeedDequeueTupleOp : TF_Op<"InfeedDequeueTuple", []> {
  let summary = "Fetches multiple values from infeed as an XLA tuple.";

  let description = [{
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$_XlaSharding
  );

  let results = (outs
    Variadic<TF_Tensor>:$outputs
  );

  TF_DerivedResultShapeListAttr shapes = TF_DerivedResultShapeListAttr<0>;
  TF_DerivedResultTypeListAttr dtypes = TF_DerivedResultTypeListAttr<0>;
}

def TF_StringFormatOp : TF_Op<"StringFormat", [NoSideEffect]> {
  let summary = "Formats a string template using a list of tensors.";

  let description = [{
Formats a string template using a list of tensors, pretty-printing tensor summaries.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,

    DefaultValuedAttr<StrAttr, "%s">:$strtemplate,
    DefaultValuedAttr<StrAttr, "%s">:$placeholder,
    DefaultValuedAttr<I64Attr, "3">:$summarize
  );

  let results = (outs
    TF_StrTensor:$output
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
}

//===----------------------------------------------------------------------===//
// tf.data ops
//===----------------------------------------------------------------------===//

def TF_BatchDatasetV2Op : TF_Op<"BatchDatasetV2", [NoSideEffect]> {
  let summary = [{
Creates a dataset that batches `batch_size` elements from `input_dataset`.
  }];

  let description = [{
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    I64Tensor:$batch_size,
    I1Tensor:$drop_remainder,

    DefaultValuedAttr<BoolAttr, "false">:$parallel_copy,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_MapDatasetOp : TF_Op<"MapDataset", [NoSideEffect]> {
  let summary = [{
    Creates a dataset that applies `f` to the outputs of `input_dataset`.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "true">:$use_inter_op_parallelism,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_MapAndBatchDatasetOp : TF_Op<"MapAndBatchDataset", [NoSideEffect]> {
  let summary = "Creates a dataset that fuses mapping with batching.";

  let description = [{
Creates a dataset that applies `f` to the outputs of `input_dataset` and then
batches `batch_size` of them.

Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `batch_size * num_parallel_batches` copies of `f` in parallel.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,
    I64Tensor:$batch_size,
    I64Tensor:$num_parallel_calls,
    I1Tensor:$drop_remainder,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_ParallelMapDatasetOp : TF_Op<"ParallelMapDataset", [NoSideEffect]> {
  let summary = [{
    Creates a dataset that applies `f` to the outputs of `input_dataset`.
  }];

  let description = [{
    Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes
    up to `num_parallel_calls` copies of `f` in parallel.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,
    I32Tensor:$num_parallel_calls,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "true">:$use_inter_op_parallelism,
    DefaultValuedAttr<BoolAttr, "false">:$sloppy,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_TensorSliceDatasetOp : TF_Op<"TensorSliceDataset", []> {
  let summary = [{
    Creates a dataset that emits each dim-0 slice of `components` once.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$components,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Toutput_types = TF_DerivedOperandTypeListAttr<0>;
}

// TODO(b/156507832): Move tf.InplaceUpdate to tf_generated_ops.td once
// autogenerated op def matches.
def TF_InplaceUpdateOp : TF_Op<"InplaceUpdate", [NoSideEffect]> {
  let summary = "Updates specified rows 'i' with values 'v'.";

  let description = [{
Computes `x[i, :] = v; return x`.

Originally this function is mutative however for compilation we make this
operation create / operate on a copy of `x`.
  }];

  let arguments = (ins
    TF_Tensor:$x,
    I32Tensor:$i,
    TF_Tensor:$v
  );

  let results = (outs
    TF_Tensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

#endif // TF_OPS
