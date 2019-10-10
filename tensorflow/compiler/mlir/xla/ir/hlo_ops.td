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

// This is the operation definition file for XLA.

#ifdef HLO_OPS
#else
#define HLO_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef HLO_OPS_BASE
#else
include "tensorflow/compiler/mlir/xla/ir/hlo_ops_base.td"
#endif

def HLO_Dialect : Dialect {
  let name = "xla_hlo";
  let cppNamespace = "xla_hlo";
}

class HLO_Op<string mnemonic, list<OpTrait> traits> :
    Op<HLO_Dialect, mnemonic, traits> {
  // Whether this operation has a custom conversion to HLO or not.
  bit hasCustomHLOConverter = 0b0;

  // TODO(b/129012527) Much of this custom verification should be expressed as
  // type constraints.
  let verifier = [{ return Verify(*this); }];
}

//===----------------------------------------------------------------------===//
// XLA type definitions.
//===----------------------------------------------------------------------===//

// Any integer tensor types
def HLO_IntTensor : TensorOf<[HLO_Int]>;

// Any floating-point tensor types
def HLO_FpTensor : TensorOf<[AnyFloat]>;

def HLO_PredTensor : TensorOf<[HLO_Pred]>;

// Any integer or floating-point tensor types
def HLO_IntOrFpTensor : TensorOf<[HLO_Int, AnyFloat]>;

def HLO_Tensor : TensorOf<[AnyFloat, AnyInteger]>;

def HLO_Tuple : NestedTupleOf<[HLO_Tensor]>;

def HLO_TensorOrTuple : AnyTypeOf<[HLO_Tensor, HLO_Tuple]>;

//===----------------------------------------------------------------------===//
// XLA nullary op definitions.
//===----------------------------------------------------------------------===//

def HLO_ConstOp : BASE_HLO_ConstOp, HLO_Op<"constant", [NoSideEffect]> {
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Attribute value"
  >];

  let printer = [{ return Print(*this, &p); }];
  let parser = [{ return ParseConstOp(&parser, &result); }];

  let hasFolder = 1;

  // Constant has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_IotaOp : BASE_HLO_IotaOp, HLO_Op<"iota", [NoSideEffect]> {
  let arguments = (ins I64Attr:$iota_dimension);

  let results = (outs HLO_Tensor:$output);

  let hasFolder = 1;

  // TODO(b/130357376): Iota has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// XLA unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions
class HLO_UnaryElementwiseOp<string mnemonic, list<OpTrait> traits>:
    HLO_Op<mnemonic, traits> {

    let arguments = (ins HLO_Tensor);
    let results = (outs HLO_Tensor);
}

def HLO_AbsOp: HLO_UnaryElementwiseOp<"abs", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_AbsOp;

def HLO_CeilOp: HLO_UnaryElementwiseOp<"ceil", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_CeilOp;

def HLO_ConvertOp : HLO_UnaryElementwiseOp<
      "convert", [NoSideEffect, SameOperandsAndResultShape]>, BASE_HLO_ConvertOp {
  let hasFolder = 1;

  // TODO(b/130357376) Convert has a special constructor. Use a custom
  // HLO converter until we have a method to call the special constructor.
  let hasCustomHLOConverter = 1;
}

def HLO_CosOp: HLO_UnaryElementwiseOp<"cos", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_CosOp;

def HLO_ExpOp: HLO_UnaryElementwiseOp<"exp", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_ExpOp;

def HLO_FloorOp: HLO_UnaryElementwiseOp<"floor", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_FloorOp;

def HLO_LogOp: HLO_UnaryElementwiseOp<"log", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_LogOp;

def HLO_NegOp: HLO_UnaryElementwiseOp<"neg", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_NegOp;

def HLO_RsqrtOp: HLO_UnaryElementwiseOp<"rsqrt", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_RsqrtOp;

def HLO_SignOp: HLO_UnaryElementwiseOp<"sign", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_SignOp;

def HLO_TanhOp: HLO_UnaryElementwiseOp<"tanh",
    [ResultsAreFloatLike, NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_TanhOp;

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class HLO_BinaryElementwiseOp<string mnemonic, list<OpTrait> traits> :
        HLO_Op<mnemonic, traits> {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    BroadcastDimAttr:$broadcast_dimensions
  );
  let results = (outs HLO_Tensor);
  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def HLO_AddOp : HLO_BinaryElementwiseOp<"add",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_AddOp;

def HLO_DivOp : HLO_BinaryElementwiseOp<"div",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_DivOp;

def HLO_MaxOp : HLO_BinaryElementwiseOp<"max",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MaxOp;

def HLO_MinOp : HLO_BinaryElementwiseOp<"min",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MinOp;

def HLO_MulOp : HLO_BinaryElementwiseOp<"mul",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MulOp;

def HLO_SubOp : HLO_BinaryElementwiseOp<"sub",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_SubOp;


//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class HLO_BinaryLogicalElementwiseOp<string mnemonic> :
        HLO_BinaryElementwiseOp<mnemonic, [Commutative, NoSideEffect]> {
  let arguments = (ins
    HLO_PredTensor:$lhs,
    HLO_PredTensor:$rhs,
    BroadcastDimAttr:$broadcast_dimensions
  );
}

def HLO_AndOp: HLO_BinaryLogicalElementwiseOp<"and">, BASE_HLO_AndOp;
def HLO_OrOp: HLO_BinaryLogicalElementwiseOp<"or">, BASE_HLO_OrOp;

//===----------------------------------------------------------------------===//
// XLA control flow op definitions.
//===----------------------------------------------------------------------===//
def HLO_WhileOp: HLO_Op<"while", [NoSideEffect, SameOperandsAndResultType]> {
  string summary = "While operator";

  string description = [{
    Returns the result of executing a body function until the cond body returns
    true.

    See https://www.tensorflow.org/xla/operation_semantics#while.
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrTuple>:$val,
    SymbolRefAttr:$cond,
    SymbolRefAttr:$body
  );

  let results = (outs Variadic<HLO_TensorOrTuple>);

  // TODO(b/129422361): WhileOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_ReduceOp: HLO_Op<"reduce", [
      NoSideEffect,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]>, BASE_HLO_ReduceOp {
  let arguments = (ins
    Variadic<HLO_TensorOrTuple>:$operands,
    Variadic<HLO_TensorOrTuple>:$init_values,
    I64ElementsAttr:$dimensions
  );

  let results = (outs Variadic<HLO_TensorOrTuple>);

  let hasFolder = 1;

  // TODO(hinsu): Verify that the attached body arguments and results are
  // compatible with reduce op's operands.
  let regions = (region SizedRegion<1>:$body);

  // TODO(hinsu): Implement custom printer and parser.

  // TODO(b/129422361): ReduceOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// XLA tuple op definitions.
//===----------------------------------------------------------------------===//
def HLO_GetTupleElementOp: HLO_Op<"get_tuple_element", [NoSideEffect]>, BASE_HLO_GetTupleElementOp {
  let arguments = (ins
    HLO_Tuple,
    I32Attr:$index
  );

  let results = (outs HLO_TensorOrTuple);

  let hasFolder = 1;

  // GetTupleElementOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_TupleOp : HLO_Op<"tuple", [NoSideEffect]>, BASE_HLO_TupleOp {
   let arguments = (ins Variadic<HLO_TensorOrTuple>:$val);
   let results = (outs HLO_Tuple);

  // TupleOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_CompareOp: HLO_Op<"compare",
      [NoSideEffect, SameOperandsElementType]>, BASE_HLO_CompareOp {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    BroadcastDimAttr:$broadcast_dimensions,
    HLO_ComparisonDirectionAttr:$comparison_direction
  );
  let results = (outs HLO_PredTensor);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *lhs, Value *rhs, "
    "DenseIntElementsAttr broadcast_dimensions, StringAttr comparison_direction"
  >];

  let extraClassDeclaration = [{
    // Infers output type for given operand and attributes.
    static Type InferOutputTypes(Builder *builder, Value *lhs, Value *rhs,
                                 DenseIntElementsAttr broadcast_dimensions,
                                 StringAttr comparison_direction);
  }];
}

//===----------------------------------------------------------------------===//
// XLA Slice definitions.
//===----------------------------------------------------------------------===//

def HLO_SliceOp: HLO_Op<
      "slice",
      [NoSideEffect, SameOperandsAndResultElementType,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>]> {
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );

  let results = (outs HLO_Tensor);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *operand, "
    "DenseIntElementsAttr start_indices, DenseIntElementsAttr limit_indices, "
    "DenseIntElementsAttr strides"
  >];

  let extraClassDeclaration = [{
    // Infers output type for given operand and attributes. Result type is
    // unranked if any of the attributes is illegal.
    static Type InferOutputTypes(Builder *builder, Value *operand,
                                 DenseIntElementsAttr start_indices,
                                 DenseIntElementsAttr limit_indices,
                                 DenseIntElementsAttr strides);
  }];

  // TODO(b/129422361) Two of the required arguments comes from the start and
  // limit indices which aren't handled by the codegen.
  let hasCustomHLOConverter = 1;
}

def HLO_DynamicSliceOp: HLO_Op<"dynamic-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "result"]>,
       AllTypesMatch<["start_indices", "slice_sizes"]>]> {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$start_indices,
    I64ElementsAttr:$slice_sizes
  );

  let results = (outs HLO_Tensor:$result);

  let hasCustomHLOConverter = 1;
}

def HLO_DynamicUpdateSliceOp: HLO_Op<"dynamic-update-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "result"]>]> {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$update,
    Variadic<HLO_Tensor>:$start_indices
  );

  let results = (outs HLO_Tensor:$result);

  // TODO(b/129422361) Requires a custom constructor.
  let hasCustomHLOConverter = 1;
}


//===----------------------------------------------------------------------===//
// XLA Other op definitions.
//===----------------------------------------------------------------------===//

def HLO_BatchNormInferenceOp : HLO_Op<"batch_norm_inference", [NoSideEffect]>,
    BASE_HLO_BatchNormInferenceOp {

  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$scale,
    HLO_Tensor:$offset,
    HLO_Tensor:$mean,
    HLO_Tensor:$variance,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs HLO_Tensor);
}

def HLO_BroadcastOp : HLO_Op<"broadcast",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_BroadcastOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$broadcast_sizes
  );

  let results = (outs HLO_Tensor);
}

def HLO_BroadcastInDimOp : HLO_Op<"broadcast_in_dim",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_BroadcastInDimOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    BroadcastDimAttr:$broadcast_dimensions
  );

  let results = (outs HLO_Tensor);

  // TODO(b/130357376): One of the arguments comes from the new shape, which is
  // not handled by the codegen.
  let hasCustomHLOConverter = 1;
}

def HLO_ClampOp : HLO_Op<"clamp",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ClampOp {
  let arguments = (ins
    HLO_Tensor:$min,
    HLO_Tensor:$operand,
    HLO_Tensor:$max
  );

  let results = (outs HLO_Tensor);
}

def HLO_ConcatenateOp : HLO_Op<"concatenate",
    [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ConcatenateOp {

  let arguments = (ins
    Variadic<HLO_Tensor>:$val,
    I64Attr: $dimension
  );

  let results = (outs HLO_Tensor);

  let hasFolder = 1;

  // TODO(b/129422361) ConcatOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

// TODO(hinsu): Make this struct dialect independent so that it can be shared
// between HLO and LHLO dialect.
def ConvDimensionNumbers : StructAttr<"ConvDimensionNumbers", HLO_Dialect, [
  StructFieldAttr<"input_batch_dimension",I64Attr>,
  StructFieldAttr<"input_feature_dimension", I64Attr>,
  StructFieldAttr<"input_spatial_dimensions", I64ElementsAttr>,
  StructFieldAttr<"kernel_input_feature_dimension", I64Attr>,
  StructFieldAttr<"kernel_output_feature_dimension", I64Attr>,
  StructFieldAttr<"kernel_spatial_dimensions", I64ElementsAttr>,
  StructFieldAttr<"output_batch_dimension", I64Attr>,
  StructFieldAttr<"output_feature_dimension", I64Attr>,
  StructFieldAttr<"output_spatial_dimensions", I64ElementsAttr>] > {

  let description = "Structure of dimension information for conv op";
}

def HLO_ConvOp : HLO_Op<"conv", [NoSideEffect]>, BASE_HLO_ConvOp {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    // Default value: zero for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$padding,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    HLO_PrecisionConfigAttr:$precision_config
  );

  let results = (outs HLO_Tensor);

  // TODO(b/129422361): Conv Op has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_CopyOp: HLO_Op<"copy", [NoSideEffect, SameOperandsAndResultType]> {
  string summary = "Copy operator";

  string description = [{
    Returns a copy of `operand`.
  }];

  let arguments = (ins HLO_Tensor);
  let results = (outs HLO_Tensor);

  // TODO(b/129422361) Implement special handling.
  // Copy has an HloOpcode, but is not one of the ops defined in xla_builder.
  let hasCustomHLOConverter = 1;
}

def HLO_DotOp: HLO_Op<"dot", [NoSideEffect]>, BASE_HLO_DotOp {
  let arguments = (
    ins HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    HLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs HLO_Tensor);
}

def DotDimensionNumbers : StructAttr<"DotDimensionNumbers", HLO_Dialect, [
                StructFieldAttr<"lhs_batching_dimensions",   ElementsAttr>,
                StructFieldAttr<"rhs_batching_dimensions",   ElementsAttr>,
                StructFieldAttr<"lhs_contracting_dimensions", ElementsAttr>,
                StructFieldAttr<"rhs_contracting_dimensions", ElementsAttr>] > {
  let description = "Structure of dimension information for dot product";
}

def HLO_DotGeneralOp: HLO_Op<"dot_general", [NoSideEffect]>, BASE_HLO_DotGeneralOp {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    DotDimensionNumbers:$dot_dimension_numbers,
    HLO_PrecisionConfigAttr:$precision_config
  );

  let results = (outs HLO_Tensor);
}

def HLO_GatherOp: HLO_Op<"gather", [NoSideEffect]>, BASE_HLO_GatherOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_IntTensor:$start_indices,
    I64Attr:$index_vector_dim,
    I64ElementsAttr:$offset_dims,
    I64ElementsAttr:$slice_sizes,
    I64ElementsAttr:$collapsed_slice_dims,
    I64ElementsAttr:$start_index_map
  );

  let results = (outs HLO_Tensor);

  // TODO(b/129422361) Attributes are not by the codegen. The optional argument
  // (dimensions) needs to be added as an attribute.
  let hasCustomHLOConverter = 1;
}

def HLO_ReshapeOp: HLO_Op<"reshape",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ReshapeOp {
  let arguments = (ins HLO_Tensor:$operand);

  let results = (outs HLO_Tensor);
  let hasFolder = 1;

  // TODO(b/129422361) One of the required arguments comes from the new shape,
  // which isn't handled by the codegen. The optional argument (dimensions)
  // needs to be added as an attribute.
  let hasCustomHLOConverter = 1;
}


def HLO_SelectOp: HLO_Op<"select", [NoSideEffect]>, BASE_HLO_SelectOp {
  let arguments = (ins
    HLO_PredTensor:$pred,
    HLO_Tensor:$on_true,
    HLO_Tensor:$on_false
  );

  let results = (outs HLO_Tensor);
}

def HLO_ReverseOp: HLO_Op<"reverse",
      [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_ReverseOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$dimensions
  );

  let results = (outs HLO_Tensor);

  let hasFolder = 1;

  // TODO(b/129422361): ReverseOp has a custom constructor for HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_PadOp: HLO_Op<"pad",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_PadOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$padding_value,
    I64ElementsAttr: $edge_padding_low,
    I64ElementsAttr: $edge_padding_high,
    I64ElementsAttr: $interior_padding
  );

  let results = (outs HLO_Tensor);

  let description = [{
    Pads the `operand` according to TBD.
  }];

  // TODO(b/129422361): PadOp has a custom constructor for HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_TransposeOp: HLO_Op<"transpose",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_TransposeOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$permutation
  );
  let results = (outs HLO_Tensor);

  let hasFolder = 1;
}

def HLO_ReduceWindowOp: HLO_Op<"reduce_window", [
      NoSideEffect,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]>, BASE_HLO_ReduceWindowOp {

  // TODO(hinsu): Verify that padding attribute is 2-d and the remaining
  // attributes are 1-d. Attributes' leading dimension should match rank of the
  // inputs.
  let arguments = (ins
    Variadic<HLO_TensorOrTuple>:$operands,
    Variadic<HLO_TensorOrTuple>:$init_values,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the input dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let results = (outs Variadic<HLO_Tensor>);

  // TODO(hinsu): Verify that the attached body arguments and results are
  // compatible with reduce op's operands.
  let regions = (region SizedRegion<1>:$body);

  // TODO(b/129422361): ReduceWindowOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  // TODO(hinsu): Implement custom printer and parser.
}

def HLO_ReturnOp : HLO_Op<"return", [Terminator]> {
  let summary = [{
    The `hlo.return` operation terminates a region and returns values.
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrTuple>:$results
  );

  // Disable conversion operator for return op as the op is not an actual XLA
  // instruction and is only used as a terminator for regions.
  let hasCustomHLOConverter = 1;

  // TODO(hinsu): Implement custom printer and parser.
}

//===----------------------------------------------------------------------===//
// XLA RngUniform Operator.
//===----------------------------------------------------------------------===//
def HLO_RngUniformOp : HLO_Op<"rng_uniform", []>, BASE_HLO_RngUniformOp {
  let arguments = (ins
    HLO_Tensor:$a,
    HLO_Tensor:$b,
    I64Tensor:$shape
  );

  let results = (outs HLO_Tensor);

  // TODO(bgogul): Disable conversion operator for `RngUniform` as
  // the default constructor for `xla::RngUniform` takes `Shape` as
  // the last argument and the default converter does not deal with it.
  let hasCustomHLOConverter = 1;
}

#endif // HLO_OPS
