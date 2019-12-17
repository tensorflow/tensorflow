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

#ifndef HLO_OPS
#define HLO_OPS

include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops_base.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_utils.td"

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

// Token type.
def HLO_Token : Type<CPred<"$_self.isa<TokenType>()">, "token">;

// Any integer tensor types
def HLO_IntTensor : TensorOf<[HLO_Int]>;

// Any floating-point tensor types
def HLO_FpTensor : TensorOf<[AnyFloat]>;

def HLO_PredTensor : TensorOf<[HLO_Pred]>;

def HLO_Tensor : TensorOf<[AnyFloat, AnyInteger, AnyComplex]>;

def HLO_ComplexTensor : TensorOf<[AnyComplex]>;

def HLO_Tuple : NestedTupleOf<[HLO_Tensor, HLO_Token]>;

def HLO_TensorOrTuple : AnyTypeOf<[HLO_Tensor, HLO_Tuple]>;

//===----------------------------------------------------------------------===//
// XLA combined type definitions.
//===----------------------------------------------------------------------===//

// Any integer or floating-point tensor types
def HLO_IntOrFpTensor : TensorOf<[HLO_Int, AnyFloat]>;

// Any integer or predicate tensor types
def HLO_PredOrIntTensor : TensorOf<[HLO_Pred, HLO_Int]>;

// Any floating-point or complex tensor types
def HLO_FpOrComplexTensor : TensorOf<[AnyFloat, AnyComplex]>;

// Any int, floating-point or complex tensor types
def HLO_IntFpOrComplexTensor : TensorOf<[HLO_Int, AnyFloat, AnyComplex]>;

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
class HLO_UnaryElementwiseOp<string mnemonic, list<OpTrait> traits,
      Type TensorType>: HLO_Op<mnemonic, traits> {

    let arguments = (ins TensorType:$operand);
    let results = (outs TensorType);
}

// Abs supports complex to real, so element type is not guaranteed to match.
def HLO_AbsOp: HLO_UnaryElementwiseOp<"abs",
    [NoSideEffect, SameOperandsAndResultShape], HLO_Tensor>, BASE_HLO_AbsOp {
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *operand"
  >];
}

def HLO_CeilOp: HLO_UnaryElementwiseOp<"ceil",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpTensor>, BASE_HLO_CeilOp;

def HLO_ConvertOp : HLO_UnaryElementwiseOp<
    "convert", [NoSideEffect, SameOperandsAndResultShape], HLO_Tensor>,
    BASE_HLO_ConvertOp {

  let builders = [OpBuilder<
    "Builder *, OperationState &tblgen_state, Value *operand, "
    "Type result_element_ty"
  >];

  let hasFolder = 1;

  let hasCustomHLOConverter = 1;
}

def HLO_CosOp: HLO_UnaryElementwiseOp<"cos",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor>,
    BASE_HLO_CosOp;

def HLO_ExpOp: HLO_UnaryElementwiseOp<"exp",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor>,
    BASE_HLO_ExpOp;

def HLO_Expm1Op: HLO_UnaryElementwiseOp<"exponential_minus_one",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor>,
    BASE_HLO_Expm1Op;

def HLO_FloorOp: HLO_UnaryElementwiseOp<"floor",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpTensor>, BASE_HLO_FloorOp;

def HLO_IsFiniteOp: HLO_UnaryElementwiseOp<"is_finite",
    [NoSideEffect, SameOperandsAndResultShape], HLO_Tensor>,
    BASE_HLO_IsFiniteOp {
  let arguments = (ins HLO_FpTensor:$x);
  let results = (outs HLO_PredTensor:$y);
}

def HLO_LogOp: HLO_UnaryElementwiseOp<"log",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor>,
    BASE_HLO_LogOp;

def HLO_Log1pOp: HLO_UnaryElementwiseOp<"log_plus_one",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor>,
    BASE_HLO_Log1pOp;

def HLO_NotOp: HLO_UnaryElementwiseOp<"not",
    [NoSideEffect, SameOperandsAndResultType], HLO_PredOrIntTensor>,
    BASE_HLO_NotOp;

def HLO_NegOp: HLO_UnaryElementwiseOp<"neg",
    [NoSideEffect, SameOperandsAndResultType], HLO_IntFpOrComplexTensor>,
    BASE_HLO_NegOp;

def HLO_PopulationCountOp: HLO_UnaryElementwiseOp<"popcnt",
    [NoSideEffect, SameOperandsAndResultType], HLO_IntTensor>,
    BASE_HLO_PopulationCountOp;

def HLO_RsqrtOp: HLO_UnaryElementwiseOp<"rsqrt",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor>,
    BASE_HLO_RsqrtOp;

def HLO_SignOp: HLO_UnaryElementwiseOp<"sign",
    [NoSideEffect, SameOperandsAndResultType], HLO_IntFpOrComplexTensor>,
    BASE_HLO_SignOp;

def HLO_SinOp: HLO_UnaryElementwiseOp<"sin",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor>,
    BASE_HLO_SinOp;

def HLO_SqrtOp: HLO_UnaryElementwiseOp<"sqrt",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor>,
    BASE_HLO_SqrtOp;

def HLO_TanhOp: HLO_UnaryElementwiseOp<"tanh",
    [ResultsAreFloatLike, NoSideEffect, SameOperandsAndResultType],
    HLO_FpOrComplexTensor>, BASE_HLO_TanhOp;

//===----------------------------------------------------------------------===//
// XLA complex unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

def HLO_ComplexOp: HLO_Op<"complex",
    [NoSideEffect, SameOperandsElementType, SameOperandsAndResultShape]>,
    BASE_HLO_ComplexOp {
  let builders = [OpBuilder<
    "Builder *, OperationState &tblgen_state, Value *lhs, Value *rhs">];

  let arguments = (ins HLO_FpTensor:$lhs, HLO_FpTensor:$rhs);
  let results = (outs HLO_ComplexTensor);
  let hasFolder = 1;
}

def HLO_ImagOp: HLO_Op<
    "imag", [NoSideEffect, SameOperandsAndResultShape]>, BASE_HLO_ImagOp {
  let builders = [OpBuilder<
    "Builder *, OperationState &tblgen_state, Value *val">];

  let arguments = (ins HLO_ComplexTensor);
  let results = (outs HLO_FpTensor);
  let hasFolder = 1;
}

def HLO_RealOp: HLO_Op<
    "real", [NoSideEffect, SameOperandsAndResultShape]>, BASE_HLO_RealOp {
  let builders = [OpBuilder<
    "Builder *, OperationState &tblgen_state, Value *val">];

  let arguments = (ins HLO_ComplexTensor);
  let results = (outs HLO_FpTensor);
  let hasFolder = 1;
}

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

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *left, Value* right, "
    "DenseIntElementsAttr broadcast_dimensions"
  >];

  let results = (outs HLO_Tensor);
  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def HLO_AddOp : HLO_BinaryElementwiseOp<"add",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_AddOp;

def HLO_Atan2Op : HLO_BinaryElementwiseOp<"atan2",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_Atan2Op;

def HLO_DivOp : HLO_BinaryElementwiseOp<"div",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_DivOp;

def HLO_MaxOp : HLO_BinaryElementwiseOp<"max",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MaxOp;

def HLO_MinOp : HLO_BinaryElementwiseOp<"min",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MinOp;

def HLO_MulOp : HLO_BinaryElementwiseOp<"mul",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MulOp;

def HLO_PowOp : HLO_BinaryElementwiseOp<"pow",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_PowOp;

def HLO_RemOp : HLO_BinaryElementwiseOp<"remainder",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_RemOp;

def HLO_ShiftLeftOp : HLO_BinaryElementwiseOp<"shift_left",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ShiftLeftOp;

def HLO_ShiftRightArithmeticOp : HLO_BinaryElementwiseOp<"shift_right_arithmetic",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ShiftRightArithmeticOp;

def HLO_ShiftRightLogicalOp : HLO_BinaryElementwiseOp<"shift_right_logical",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ShiftRightLogicalOp;

def HLO_SubOp : HLO_BinaryElementwiseOp<"sub",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_SubOp;

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class HLO_BinaryLogicalElementwiseOp<string mnemonic> :
        HLO_BinaryElementwiseOp<mnemonic, [Commutative, NoSideEffect]> {
  let arguments = (ins
    HLO_PredOrIntTensor:$lhs,
    HLO_PredOrIntTensor:$rhs,
    BroadcastDimAttr:$broadcast_dimensions
  );
}

def HLO_AndOp: HLO_BinaryLogicalElementwiseOp<"and">, BASE_HLO_AndOp;
def HLO_OrOp: HLO_BinaryLogicalElementwiseOp<"or">, BASE_HLO_OrOp;
def HLO_XorOp : HLO_BinaryLogicalElementwiseOp<"xor">, BASE_HLO_XorOp;

//===----------------------------------------------------------------------===//
// XLA communication op definitions.
//===----------------------------------------------------------------------===//

// InfeedOp corresponds to 'InfeedWithToken' xla client API and not 'Infeed'.
// InfeedWithToken allows ordering of infeed HLO instructions using tokens.
def HLO_InfeedOp : HLO_Op<"infeed", []> {

  string summary = "Infeed operator";

  string description = [{
    Reads a single data item from the implicit Infeed streaming interface of
    the device, interpreting the data as the given shape, and returns a XlaOp
    of the data. Multiple Infeed operations are allowed in a computation, but
    there must be a total order among the Infeed operations.

    See https://www.tensorflow.org/xla/operation_semantics#infeed.
  }];

  let arguments = (ins HLO_Token:$token);
  let results = (outs HLO_Tuple);
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// XLA control flow op definitions.
//===----------------------------------------------------------------------===//

def HLO_AfterAllOp : HLO_Op<"after_all", []> {

  string summary = "AfterAll operator";

  string description = [{
    AfterAll takes a variadic number of tokens and produces a single token.
    Tokens are primitive types which can be threaded between side-effecting
    operations to enforce ordering. AfterAll can be used as a join of tokens
    for ordering a operation after a set operations.

    See https://www.tensorflow.org/xla/operation_semantics#afterall.
  }];

  let arguments = (ins Variadic<HLO_Token>:$operands);
  let results = (outs HLO_Token);
  let hasCustomHLOConverter = 1;
}

def HLO_ConditionalOp: HLO_Op<"conditional", [NoSideEffect]> {
  string summary = "Conditional operator";

  string description = [{
    Returns the result of executing either a true or false function depending on
    the result of a condition function.

    See https://www.tensorflow.org/xla/operation_semantics#conditional.
  }];

  let arguments = (ins
    HLO_PredTensor:$pred,
    HLO_TensorOrTuple:$true_arg,
    HLO_TensorOrTuple:$false_arg
  );

  let regions = (region AnyRegion:$true_branch, AnyRegion:$false_branch);

  let results = (outs HLO_TensorOrTuple);

  // TODO(b/129422361): ConditionalOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_WhileOp: HLO_Op<"while", [NoSideEffect, SameOperandsAndResultType]> {
  string summary = "While operator";

  string description = [{
    Returns the result of executing a body function until the cond body returns
    true.

    See https://www.tensorflow.org/xla/operation_semantics#while.
  }];

  let arguments = (ins HLO_TensorOrTuple:$val);

  let regions = (region AnyRegion:$cond, AnyRegion:$body);

  let results = (outs HLO_TensorOrTuple);

  // TODO(b/129422361): WhileOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

// Represents a unique identifier for each Send/Recv instruction pair or
// optionally for collective instructions (AllReduce, CollectivePermute,
// AllToAll). Non-positive channel_id handle is equivalent to no channel id.
def ChannelHandle : StructAttr<"ChannelHandle", HLO_Dialect, [
                StructFieldAttr<"handle", I64Attr>,
                StructFieldAttr<"type", I64Attr>]> {
  let description = "two 64-bit integers 'handle' and 'type'";
}

def HLO_AllReduceOp : HLO_Op<"all_reduce",
    [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_AllReduceOp {

  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<ChannelHandle>:$channel_id
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs HLO_Tensor);

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

  let builders = [OpBuilder<
    "Builder *, OperationState &state, ValueRange operands, "
    "ValueRange init_values, DenseIntElementsAttr dimensions"
  >];

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

  let builders = [OpBuilder<
                  "Builder *builder, OperationState &results, "
                  "Value* value, int32_t index">];
}

def HLO_TupleOp : HLO_Op<"tuple", [NoSideEffect]>, BASE_HLO_TupleOp {
  let arguments = (ins Variadic<HLO_TensorOrTuple>:$val);
  let results = (outs HLO_Tuple);

  let builders = [OpBuilder<
                  "Builder *builder, OperationState &results, "
                  "ValueRange values">];

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
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *left, Value* right, "
    "DenseIntElementsAttr broadcast_dimensions, "
    "StringAttr comparison_direction"
  >];
  let results = (outs HLO_PredTensor);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *lhs, Value *rhs, "
    "DenseIntElementsAttr broadcast_dimensions, StringAttr comparison_direction"
  >];
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
  let hasCanonicalizer = 1;
}

def HLO_DynamicUpdateSliceOp: HLO_Op<"dynamic-update-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "result"]>]> {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$update,
    Variadic<HLO_Tensor>:$start_indices
  );

  let results = (outs HLO_Tensor:$result);
}


//===----------------------------------------------------------------------===//
// XLA Other op definitions.
//===----------------------------------------------------------------------===//

def HLO_BatchNormGradOp : HLO_Op<"batch_norm_grad", [NoSideEffect]>,
    BASE_HLO_BatchNormGradOp {

  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$scale,
    HLO_Tensor:$mean,
    HLO_Tensor:$variance,
    HLO_Tensor:$grad_output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs HLO_Tuple);
}

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

def HLO_BatchNormTrainingOp : HLO_Op<"batch_norm_training", [NoSideEffect]>,
    BASE_HLO_BatchNormTrainingOp {

  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$scale,
    HLO_Tensor:$offset,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs HLO_Tuple);
}

def HLO_BitcastConvertOp : HLO_Op<"bitcast_convert",
    [NoSideEffect, SameOperandsAndResultShape]>, BASE_HLO_BitcastConvertOp {

  let arguments = (ins HLO_Tensor:$operand);
  let results = (outs HLO_Tensor);
  let hasCustomHLOConverter = 1;
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

def HLO_CrossReplicaSumOp : HLO_Op<"cross-replica-sum",
    [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_CrossReplicaSumOp {

  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups
  );

  let results = (outs HLO_Tensor);
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

def BASE_EinsumOp {
  string summary = "Einsum operator";

  string description = [{
    Returns a tensor whose elements are defined by equation, which is written
    in a shorthand form inspired by the Einstein summation convention.
  }];
}

def HLO_EinsumOp: HLO_Op<"einsum", [NoSideEffect]> {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    StrAttr:$einsum_config
  );

  let results = (outs HLO_Tensor);

  // TODO(hinsu): Canonicalize to lower this client side HLO op to server
  // side HLO ops.
}

def HLO_UnaryEinsumOp: HLO_Op<"unary_einsum", [NoSideEffect]> {
  let arguments = (ins
    HLO_Tensor:$operand,
    StrAttr:$einsum_config
  );

  let results = (outs HLO_Tensor);

  let hasCanonicalizer = 1;

  // UnaryEinsumOp is unconditionally canonicalized to the binary EinsumOp so
  // the HLO converter shouldn't be invoked.
  let hasCustomHLOConverter = 1;
}

def HLO_FftOp: HLO_Op<"fft", [NoSideEffect]>, BASE_HLO_FftOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_FftTypeAttr: $fft_type,
    I64ElementsAttr:$fft_length
  );

  let results = (outs HLO_Tensor);

  // TODO(b/129422361) Attributes are not supported by the codegen.
  let hasCustomHLOConverter = 1;
}

def GatherDimensionNumbers : StructAttr<"GatherDimensionNumbers", HLO_Dialect,
      [StructFieldAttr<"offset_dims", I64ElementsAttr>,
      StructFieldAttr<"collapsed_slice_dims", I64ElementsAttr>,
      StructFieldAttr<"start_index_map", I64ElementsAttr>,
      StructFieldAttr<"index_vector_dim", I64Attr>]> {
  let description = "Structure of dimension information for gather";
}

def HLO_GatherOp: HLO_Op<"gather", [NoSideEffect]>, BASE_HLO_GatherOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_IntTensor:$start_indices,
    GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    DefaultValuedAttr<BoolAttr, "false">:$indices_are_sorted
  );

  let results = (outs HLO_Tensor);

  let hasCustomHLOConverter = 1;
}

def HLO_GetDimensionSizeOp: HLO_Op<"get_dimension_size", [NoSideEffect]>,
      BASE_HLO_GetDimensionSizeOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    I32Attr:$dimension
  );
  let results = (outs HLO_IntTensor);
}

def HLO_ReshapeOp: HLO_Op<"reshape",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ReshapeOp {
  let arguments = (ins HLO_Tensor:$operand);

  let results = (outs HLO_Tensor);
  let hasFolder = 1;

  let hasCustomHLOConverter = 1;
}

def ScatterDimensionNumbers : StructAttr<"ScatterDimensionNumbers", HLO_Dialect,
      [StructFieldAttr<"update_window_dims", I64ElementsAttr>,
      StructFieldAttr<"inserted_window_dims", I64ElementsAttr>,
      StructFieldAttr<"scatter_dims_to_operand_dims", I64ElementsAttr>,
      StructFieldAttr<"index_vector_dim", I64Attr>]> {
  let description = "Structure of dimension information for scatter";
}

def HLO_ScatterOp: HLO_Op<"scatter", [NoSideEffect]>, BASE_HLO_ScatterOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$scatter_indices,
    HLO_Tensor:$updates,
    ScatterDimensionNumbers:$scatter_dimension_numbers,
    DefaultValuedAttr<BoolAttr, "false">:$indices_are_sorted,
    DefaultValuedAttr<BoolAttr, "false">:$unique_indices
  );

  let regions = (region SizedRegion<1>:$update_computation);

  let results = (outs HLO_Tensor);

  let hasCustomHLOConverter = 1;
}

// TODO(jpienaar): Add broadcastable trait.
def HLO_SelectOp: HLO_Op<"select", [NoSideEffect]>, BASE_HLO_SelectOp {
  let arguments = (ins
    HLO_PredTensor:$pred,
    HLO_Tensor:$on_true,
    HLO_Tensor:$on_false
  );

  let results = (outs HLO_Tensor);
}

def HLO_SelectAndScatterOp: HLO_Op<"select_and_scatter",
      [NoSideEffect]>, BASE_HLO_SelectAndScatterOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$source,
    HLO_Tensor:$init_value,
    OptionalAttr<I64ElementsAttr>:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let regions = (region SizedRegion<1>:$select, SizedRegion<1>:$scatter);

  let results = (outs HLO_Tensor);

  let hasCustomHLOConverter = 1;
}

def HLO_SortOp : HLO_Op<"sort", [NoSideEffect]>, BASE_HLO_SortOp {
  let arguments = (ins
    Variadic<HLO_Tensor>:$operands,
    DefaultValuedAttr<I64Attr, "-1">:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$is_stable
  );

  let results = (outs HLO_TensorOrTuple);

  let regions = (region SizedRegion<1>:$comparator);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &state, ValueRange operands, "
    "int64_t dimension, bool is_stable"
  >];

  // TODO(b/129422361): SortOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
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
      SingleBlockImplicitTerminator<"ReturnOp">
    ]>, BASE_HLO_ReduceWindowOp {

  // TODO(hinsu): Verify that padding attribute is 2-d and the remaining
  // attributes are 1-d. Attributes' leading dimension should match rank of the
  // inputs.
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$init_value,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the input dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let results = (outs HLO_Tensor);

  // TODO(hinsu): Verify that the attached body arguments and results are
  // compatible with reduce op's operands.
  let regions = (region SizedRegion<1>:$body);

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

def HLO_TorchIndexSelectOp : HLO_Op<"torch_index_select", [NoSideEffect]> {
  let arguments = (ins
    HLO_Tensor:$input,
    HLO_Tensor:$index,
    I64Attr:$dim,
    I64Attr:$batch_dims
  );

  let results = (outs HLO_Tensor);

  // TODO(hinsu): Canonicalize to lower this client side HLO op to server
  // side HLO ops.
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

  let hasCustomHLOConverter = 1;
}

#endif // HLO_OPS
