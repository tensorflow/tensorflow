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

// This is the operation definition file for TensorFlow Lite.

#ifndef TFL_OPS
#define TFL_OPS

include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_interfaces.td"
include "tensorflow/compiler/mlir/lite/quantization/quantization.td"

def TFL_Dialect : Dialect {
  let name = "tfl";

  let description = [{
    The TensorFlow Lite dialect.

    This dialect maps to TensorFlow Lite operations.

    Invariants:

    * All values are of Tensor type (in particular, scalars are
      represented using zero-dimensional tensors);
  }];

  let cppNamespace = "TFL";
}

//===----------------------------------------------------------------------===//
// TFLite dialect string type - uses the TF string type as implementation
//===----------------------------------------------------------------------===//
def TFL_Str : Type<CPred<"$_self.isa<mlir::TF::StringType>()">,
                  "TFLite string type">,
             BuildableType<"getType<mlir::TF::StringType>()">;

//===----------------------------------------------------------------------===//
// TFLite dialect uint8 type - uses the TF uint8 type as implementation
//===----------------------------------------------------------------------===//
def TFL_Uint8 : Type<CPred<"$_self.isa<mlir::TF::Uint8Type>()">,
                    "TFLite uint8 type">,
              BuildableType<"getType<mlir::TF::Uint8Type>()">;

//===----------------------------------------------------------------------===//
// TFLite dialect quint8 type - uses the TF quint8 type as implementation
//===----------------------------------------------------------------------===//
def TFL_Quint8 : Type<CPred<"$_self.isa<mlir::TF::Quint8Type>()">,
                    "TFLite quint8 type">,
              BuildableType<"getType<mlir::TF::Quint8Type>()">;

//===----------------------------------------------------------------------===//
// Activation function enum definitions.
//===----------------------------------------------------------------------===//

// Allowed activation function cases
// These should match the ActivationFunctionType enum in TFLite schema.
def TFL_AF_None  : StrEnumAttrCase<"NONE">;
def TFL_AF_Relu  : StrEnumAttrCase<"RELU">;
def TFL_AF_Relu1 : StrEnumAttrCase<"RELU_N1_TO_1">;
def TFL_AF_Relu6 : StrEnumAttrCase<"RELU6">;
def TFL_AF_Tanh  : StrEnumAttrCase<"TANH">;
def TFL_AF_Sign  : StrEnumAttrCase<"SIGN_BIT">;

def TFL_AFAttr : StrEnumAttr<
    "ActivationFunctionType", "fused activation enum", [
      TFL_AF_None,  TFL_AF_Relu, TFL_AF_Relu1,
      TFL_AF_Relu6, TFL_AF_Tanh, TFL_AF_Sign
    ]>;

//===----------------------------------------------------------------------===//
// Padding enum definitions.
//===----------------------------------------------------------------------===//

// Allowed padding cases
// These should match the padding enum in TFLite schema.
def TFL_PAD_Same  : StrEnumAttrCase<"SAME">;
def TFL_PAD_Valid : StrEnumAttrCase<"VALID">;
def TFL_MIRRORPAD_Reflect : StrEnumAttrCase<"REFLECT">;
def TFL_MIRRORPAD_Symmetric : StrEnumAttrCase<"SYMMETRIC">;

def TFL_PaddingAttr : StrEnumAttr<"Padding", "padding enum", [
      TFL_PAD_Same, TFL_PAD_Valid
    ]>;

def TFL_MirrorPaddingAttr : StrEnumAttr<"Padding", "Mirror pad enum", [
      TFL_MIRRORPAD_Reflect, TFL_MIRRORPAD_Symmetric
    ]>;

//===----------------------------------------------------------------------===//
// TensorType attribute definitions.
//===----------------------------------------------------------------------===//
// A type attribute containing the TensorType.
def TensorTypeAttr : TypeAttrBase<"TensorType", "Tensor type attribute">;

//===----------------------------------------------------------------------===//
// Derived shape attribute class.
//===----------------------------------------------------------------------===//
class DerivedShapeAttr<code body> : DerivedAttr<"ArrayRef<int64_t>", body>;
class DerivedTFLiteTypeAttr<code body> : DerivedAttr<"tflite::TensorType", body>;

def TFL_Int32Or64 : IntOfWidths<[32, 64]>;

def TFL_FpTensor : TensorOf<[AnyFloat]>;

def TFL_I32OrI64Tensor : TensorOf<[TFL_Int32Or64]>;

def TFL_BoolTensor : TypeAlias<I1Tensor>;

// TODO(jpienaar): Expand to all int types.
def TFL_IntTensor : TypeAlias<I32Tensor, "tensor of any integer type">;

// This is used to represent the type of "ref tensors" or tensors that are
// used as variables to track state.
def TFL_StatefulTensor : TypeAlias<AnyTensor, "stateful tensor">;

// Tensor or None type.
class TFL_TensorOfOrNone<list<Type> allowedTypes, string description = ""> :
  AnyTypeOf<[TensorOf<allowedTypes>, NoneType], description>;

def TFL_FpOrI32OrI64Tensor : TensorOf<[AnyFloat, TFL_Int32Or64]>;

//===----------------------------------------------------------------------===//
// Rank/Shape helpers.
//===----------------------------------------------------------------------===//

class TFL_OperandIsUnrankedPred<int n> :
  CPred<"$_op.getOperand(" # n # ").getType().isa<UnrankedTensorType>()">;

// TODO: Some of these could be generalized and/or moved to more general
// location.
// Returns true if the n-th operand has unknown rank or has rank m.
class TFL_OperandHasRank<int n, int m> :
  PredOpTrait<"operand " # n # " is " # m # "-D",
    Or<[TFL_OperandIsUnrankedPred<n>,
      CPred<"$_op.getOperand(" # n #
      ").getType().cast<ShapedType>().getRank() == " # m>]>>;

// Returns true if the n-th operand is ranked and has rank dim.
class TFL_OperandHasKnownRank<int n, int dim> : And<[
  CPred<"$_op.getOperand(" # n # ").getType().isa<RankedTensorType>()">,
  CPred<"$_op.getOperand(" # n # ").getType().cast<ShapedType>().getRank() == "
    # dim>]>;

// True if operand n is ranked and has a rank > dim.
class TFL_OperandIsRankedAndHasDimPred<int n, int dim> : And<[
  CPred<"$_op.getOperand(" # n # ").getType().isa<RankedTensorType>()">,
  CPred<"$_op.getOperand(" # n # ").getType().cast<ShapedType>().getRank() > "
  # dim>]>;

class TFL_OperandDimEquals<int n, int dim, int size> : And<[
  TFL_OperandIsRankedAndHasDimPred<n, dim>,
  CPred<"$_op.getOperand(" # n # ").getType().cast<ShapedType>()"
      ".getShape()[" # dim # " ] == " # size>]>;

// Returns true if the n-th operand has unknown rank or at least rank m.
class TFL_OperandHasAtleastRank<int n, int m> :
  PredOpTrait<"operand " # n # " is " # m # "-D",
    Or<[CPred<"$_op.getOperand(" # n # ").getType().isa<UnrankedTensorType>()">,
      CPred<"$_op.getOperand(" # n #
        ").getType().cast<ShapedType>().getRank() >= " # m>]>>;

class TFL_OperandRankEquals1DimOfOperand<int x, int y> :
  PredOpTrait<"operand " # x # "'s rank equals operand " # y # "'s size",
    CPred<"$_op.getOperand(" # x #
      ").getType().cast<ShapedType>().getRank() == "
      "$_op.getOperand(" # y #
      ").getType().cast<ShapedType>().getShape()[0]">>;

class TFL_Operand0DOr1ElementTensor<int x> :
  PredOpTrait<"operand #" # x # " is an 0-d tensor or 1-d tensor w/ 1 element",
    Or<[TFL_OperandHasKnownRank<x, 0>,
        And<[TFL_OperandHasKnownRank<x, 1>, TFL_OperandDimEquals<x, 0, 1>]>]>>;

// tf.uint8 and tf.quint8 are mapped to the same tflite types, so they are equal
// when used as element types.
class TFL_TFTypesWithSameBits<int i, int j, int num> :
  And<[
    Or<[CPred<"getElementTypeOrSelf($_op.getResult(" # i # ")).isa<mlir::TF::Quint" # num # "Type>()">,
        CPred<"getElementTypeOrSelf($_op.getResult(" # i # ")).isa<mlir::TF::Uint" # num # "Type>()">]>,
    Or<[CPred<"getElementTypeOrSelf($_op.getOperand(" # j # ")).isa<mlir::TF::Quint" # num # "Type>()">,
        CPred<"getElementTypeOrSelf($_op.getOperand(" # j # ")).isa<mlir::TF::Uint" # num # "Type>()">]>]>;

class TFL_OperandHasRankLessThan<int n, int m> :
  PredOpTrait<"operand " # n # " is maximum " # m # "-D",
    Or<[TFL_OperandIsUnrankedPred<n>,
      CPred<"$_op.getOperand(" # n #
      ").getType().cast<ShapedType>().getRank() <= " # m>]>>;

// This is a quantization-aware version of TCresVTEtIsSameAsOp
class TFL_TCresVTEtIsSameAsOp<int i, int j> : And<[
  TCOpResIsShapedTypePred<i, j>,
  Or<[
    TCresVTEtIsSameAsOpBase<i, j>,
    TFL_TFTypesWithSameBits<i, j, 8>,
    And<[
      SubstLeaves<"$_self", "getElementTypeOrSelf($_op.getOperand(" # j # "))",
        quant_QuantizedType.predicate>,
      CPred<"quant::QuantizedType::castToStorageType("
                "getElementTypeOrSelf($_op.getResult(" # i # "))) == "
            "quant::QuantizedType::castToStorageType("
                "getElementTypeOrSelf($_op.getOperand(" # j # ")))">]>]>]>;
//===----------------------------------------------------------------------===//
// TFL op common constraints.
//===----------------------------------------------------------------------===//

// This is a constraint for most of the binary ops, e.g., add, mul, div, etc.
// Binary ops lhs & rhs should have the same value type.
def BinaryOpSameElementTypeConstraint :
  PredOpTrait<"operands have same element type", TCopVTEtIsSameAs<0, 1>>;

//===----------------------------------------------------------------------===//
// TFL common builders.
//===----------------------------------------------------------------------===//

def TFL_BroadcastableBinaryBuilder : OpBuilder<
  "Builder *builder, OperationState &result, Value lhs, Value rhs",
  [{
    auto resultType =
      OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType());
    if (!resultType)
      mlir::emitError(result.location, "non-broadcastable operands");
    result.addOperands({lhs, rhs});
    result.types.push_back(resultType);
  }]>;

def TFL_FusedBroadcastableBinaryBuilder : OpBuilder<
  "Builder *builder, OperationState &result, Value lhs, Value rhs, "
  "StringAttr fusedActivationFunction",
  [{
    buildFusedBroadcastableBinOp(
       builder, result, lhs, rhs, fusedActivationFunction);
  }]>;

def TFL_ComparisonBinaryBuilder : OpBuilder<
  "Builder *builder, OperationState &result, Value lhs, Value rhs",
  [{
    buildComparisonBinOp(builder, result, lhs, rhs);
  }]>;

//===----------------------------------------------------------------------===//
// TFL op base class.
//===----------------------------------------------------------------------===//

class TFL_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TFL_Dialect, mnemonic, traits> {
  // FlatBuffer generation specific information.
  // -------------------------------------------
  // When generating the FlatBuffer output some operations have
  // Options (as defined in the schema). These options are effectively
  // the attributes of the operations (e.g., what padding is to be used
  // for a pooling operator). Not all operations have Options and some
  // operations share Options. The following attributes indicate whether
  // the operation has Options in the serialized FlatBuffer.

  // Whether the TFLite operator has options in the schema representation.
  bit hasOptions = 0b0;

  // Use to specify a custom options type for TFLite operators where
  // the option's name does not match the TFLite operator's name.
  // If no customOption is specified then <name>Options is used if the op
  // hasOptions.
  string customOption = ?;
}

class TFL_ConvOp<string mnemonic, string opSummary, int index> :
    TFL_Op<mnemonic, [NoSideEffect, AccumulatorUniformScale<2, 0, 1>,
    TFL_ChannelDimIndexInterface, AffineOpCoefficient<index, 1>]> {
  let summary = opSummary # " operator";

  let description = [{
    Performs convolution operation on inputs.

    Inputs:
      `inputs[0]`: required: the input activation tensor
      `inputs[1]`: required: the filter weight tensor
      `inputs[2]`: optional: the bias tensor
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TFL_TensorOfOrNone<[AnyType]>:$bias,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    TFL_AFAttr:$fused_activation_function,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w
  );

  let results = (outs AnyTensor:$output);

  let hasOptions = 0b1;
}


//===----------------------------------------------------------------------===//
// TFL op definitions.
//===----------------------------------------------------------------------===//
def TFL_AbsOp : TFL_Op<"abs", [
    NoSideEffect, SameOperandsAndResultType, NoQuantizableResult]> {
  let summary = "Absolute value operator";

  let description = [{
Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasFolder = 1;
}

def TFL_AddOp : TFL_Op<"add", [ResultsBroadcastableShape, NoSideEffect, Commutative]> {
  let summary = "Addition operator";

  let description = [{
    Element-wise addition operation.
  }];

  let arguments = (
    ins AnyTensor:$lhs,
    AnyTensor:$rhs,
    TFL_AFAttr:$fused_activation_function);

  let results = (outs AnyTensor:$output);

  let hasFolder = 1;

  let builders = [TFL_FusedBroadcastableBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];

  let hasOptions = 1;
}

def TFL_AddNOp : TFL_Op<"add_n", [Commutative, NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "add_n operator";

  let description = [{
    Adds all input tensors element-wise.
  }];

  let arguments = (ins
    Variadic<TensorOf<[F32, I32, QI16, QUI16]>>:$inputs
  );

  let results = (outs
    TensorOf<[F32, I32, QI16, QUI16]>:$sum
  );
}

def TFL_ReduceAnyOp : TFL_Op<"reduce_any", [NoSideEffect]> {
  let summary = [{
Computes the "logical or" of elements across dimensions of a tensor.
  }];

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    I1Tensor:$input,
    I32Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    I1Tensor:$output
  );

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_TransposeConvOp:
    TFL_Op<"transpose_conv", [NoSideEffect]> {
  let summary = "Transpose convolution operator";

  let description = [{
    Performs transpose convolution operation on input.
  }];

  let arguments = (
    ins 1DTensorOf<[I32]>:$output_shape,
    TensorOf<[F32, TFL_Uint8, QI8, QUI8]>:$weights,
    TensorOf<[F32, TFL_Uint8, QI8, QUI8]>:$input,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w
  );

  let results = (outs AnyTensor:$output);

  let hasOptions = 1;

  let verifier = [{ return Verify(*this); }];
}

def TFL_Convolution2DTransposeBiasOp :
  Op<TFL_Dialect, "convolution_2d_transpose_bias", [NoSideEffect]> {
  let summary = " Transpose convolution with bias operator";

  let description = [{
Performs transpose convolution operation on inputs,
with the option of adding a bias.
Note this is a custom op that is not supported in the standard runtime.

    Inputs:
      `inputs[0]`: required: the input activation tensor
      `inputs[1]`: required: the filter weight tensor
      `inputs[2]`: optional: the bias tensor
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TFL_TensorOfOrNone<[AnyType]>:$bias,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w
  );

  let results = (outs AnyTensor:$output);
}

def TFL_AveragePool2DOp:
    TFL_Op<"average_pool_2d", [NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Average_pool_2d operator";

  let description = [{
    Performs average-pooling operation on input.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$filter_height,
    I32Attr:$filter_width,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs AnyTensor:$output);

  let hasOptions = 1;
  let customOption = "Pool2DOptions";
}

def TFL_ArgMaxOp : TFL_Op<"arg_max", [NoSideEffect]> {
  let summary = "ArgMax operator";

  let description = [{
    Returns the index with the largest value across dimensions of a tensor.
  }];

  let arguments = (
    ins TensorOf<[F32, I32, I8, TFL_Uint8, QI8, QUI8]>:$input,
    TFL_I32OrI64Tensor:$dim
  );

  let results = (outs
    TFL_I32OrI64Tensor:$output
  );

  let hasOptions = 1;

  DerivedTFLiteTypeAttr output_type = DerivedTFLiteTypeAttr<[{
    return getResult().getType().cast<TensorType>().getElementType().
        cast<IntegerType>().getWidth() > 32 ? tflite::TensorType_INT64 :
            tflite::TensorType_INT32;
    }]>;
}

def TFL_ArgMinOp : TFL_Op<"arg_min", [NoSideEffect]> {
  let summary = "ArgMin operator";

  let description = [{
    Returns the index with the smallest value across dimensions of a tensor."
      a = [1, 10, 26.9, 2.8, 166.32, 62.3]
      b = tf.math.argmin(input = a)
      c = tf.keras.backend.eval(b)
  }];

  let arguments = (
    ins TensorOf<[F32, I32, I8, TFL_Uint8, QI8, QUI8]>:$input,
    TFL_I32OrI64Tensor:$dim
  );

  let results = (outs
    TFL_I32OrI64Tensor:$output
  );

  let hasOptions = 1;

  DerivedTFLiteTypeAttr output_type = DerivedTFLiteTypeAttr<[{
    return getResult().getType().cast<TensorType>().getElementType().
        cast<IntegerType>().getWidth() > 32 ? tflite::TensorType_INT64 :
            tflite::TensorType_INT32;
    }]>;
}

def TFL_CeilOp: TFL_Op<"ceil", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Ceil operator";

  let description = [{
    Returns element-wise ceil value of the input.
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);
}

def TFL_ConcatenationOp : TFL_Op<"concatenation",
  [
    NoSideEffect,
    PredOpTrait<"values and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    SameOperandsAndResultsScale
  ]> {
  let summary = "Concatenation operator";

  let description = [{
    Concatenates tensors along one dimension
  }];

  let arguments = (
    ins Variadic<TensorOf<
      [F32, I64, I32, I16, I8, QI8, QUI8, TFL_Uint8]>>:$values,
    I32Attr:$axis,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs
    TensorOf<
      [F32, I64, I32, I16, I8, QI8, QUI8, TFL_Uint8]>:$output
  );

  let hasOptions = 1;

  let hasFolder = 1;

  let verifier = [{ return Verify(*this); }];
}

def TFL_ConstOp : Op<TFL_Dialect, "pseudo_const", [NoSideEffect,
    FirstAttrDerivedResultType]> {
  let summary = "Constant pseudo op.";

  let description = [{
    Represents a constant value in TensorFlow Lite dialect. This is not an
    actual operation and it will be lowered to buffer instead.

    The op is allowed to have all the same type of attributes as tf.Const does
    (e.g., opaque TF attributes are allowed).
  }];

  let arguments = (ins ElementsAttr:$value);

  let results = (outs AnyTensor:$output);

  let hasFolder = 1;

  let builders = [OpBuilder<
    "Builder *, OperationState &state, Attribute value",
    [{
      state.addAttribute("value", value);
      state.addTypes(value.getType());
    }]>
  ];
}

def TFL_ExternalConstOp : Op<TFL_Dialect, "external_const", [NoSideEffect]> {
  let summary = "External const op.";

  let description = [{
    External const op holds a `buffer_index` which points to a constant
    in the flatbuffer.
  }];

  let arguments = (ins I32Attr:$buffer_index);

  let results = (outs AnyTensor:$output);
}

def TFL_Conv2DOp : TFL_ConvOp<"conv_2d", "Convolution", 0> {
  let extraClassDeclaration = [{
    // StatefulOpInterface:
    int GetChannelDimIndex() { return 0; }
  }];
}

def TFL_CosOp: TFL_Op<"cos", [
    NoSideEffect, SameOperandsAndResultType, NoQuantizableResult]> {
  let summary = "Cosine operator";

  let description = [{
    Computes element-wise Cosine of input
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasFolder = 1;
}

def TFL_DepthwiseConv2DOp :
    TFL_ConvOp<"depthwise_conv_2d", "Depthwise-separable convolution", 3> {
  let arguments = !con(TFL_Conv2DOp.arguments, (ins I32Attr:$depth_multiplier));

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    int GetChannelDimIndex() { return 3; }
  }];
}

def TFL_FCWO_Default  : StrEnumAttrCase<"DEFAULT">;
def TFL_FCWO_Shuffled4x16i8  : StrEnumAttrCase<"SHUFFLED4x16INT8">;

def TFL_FullyConnectedOptionsWeightFormatAttr :
    StrEnumAttr<"FullyConnectedOptionsWeightsFormat",
                "fully connected options weights format", [
      TFL_FCWO_Default, TFL_FCWO_Shuffled4x16i8
    ]>;

// TODO(jpienaar): Update post discussion on semantics of FC OP.
def TFL_FullyConnectedOp : TFL_Op<"fully_connected", [
    NoSideEffect, AccumulatorUniformScale<2, 0, 1>,
    TFL_ChannelDimIndexInterface,
    AffineOpCoefficient<-1, 1>]> {
  let summary = "Fully connected op";

  let arguments = (ins
    TensorOf<[F32, QI8, QUI8, QI16, QUI16]>:$input,
    TensorOf<[F32, QI8, QUI8, QI16, QUI16]>:$filter,
    TFL_TensorOfOrNone<[F32, QI32, QUI32]>:$bias,

    TFL_AFAttr:$fused_activation_function,
    TFL_FullyConnectedOptionsWeightFormatAttr:$weights_format,
    BoolAttr:$keep_num_dims
  );

  // Depending on the weights format, this op can have one or two outputs.
  let results = (outs
    Variadic<TensorOf<[F32, QI8, QUI8, QI16, QUI16]>>:$output
  );

  let verifier = [{ return Verify(*this); }];

  let hasOptions = 1;

  let extraClassDeclaration = [{
    // ChannelDimIndexInterface:
    int GetChannelDimIndex() { return 0; }
  }];
}

def TFL_GatherOp : TFL_Op<"gather", [
    NoSideEffect,
    SameOperandsAndResultsScale,
    TFL_OperandHasAtleastRank<0, 1>,
    PredOpTrait<"params and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>
  ]> {
  let summary = "Gather operator";

  let description = [{
    Gather slices from `params` axis `axis` according to `indices`.
  }];

  let arguments = (ins
    TensorOf<[F32, I1, I8, I32, I64, TFL_Str, QI8, QUI8, QI16]>:$params,
    TensorOf<[I32, I64]>:$indices,
    I32Attr:$axis
  );

  let builders =
  [
    OpBuilder<"Builder *builder, OperationState &result, "
      "Value params, Value indices, IntegerAttr axis",
        [{ BuildGatherOp(builder, result, params, indices, axis); }]>
  ];

  let results = (outs
    TensorOf<[F32, I1, I8, I32, I64, TFL_Str, QI8, QUI8, QI16]>:$output
  );

  let hasOptions = 1;
}

def TFL_GatherNdOp : TFL_Op<"gather_nd", [NoSideEffect]> {
  let summary = "Gather_nd operator";

  let description = [{
    Gather slices from `params` into a Tensor with shape specified by `indices`.
  }];

  let arguments = (ins
    TensorOf<[F32, I8, I64, I32, TFL_Uint8]>:$params,
    TFL_I32OrI64Tensor:$indices
  );

  let results = (outs
    TensorOf<[F32, I8, I64, I32, TFL_Uint8]>:$output
  );
}

// Same type check of lhs and rhs is handled by the ResultsBroadcastableShape trait.
def TFL_LessEqualOp : TFL_Op<"less_equal", [
    ResultsBroadcastableShape, NoSideEffect, NoQuantizableResult]> {
  let summary = "Less_equal operator";

  let description = [{
    Element-wise less_equal operation.
  }];

  let arguments = (
      ins TensorOf<[F32, I32, I64, I8, QI8, QUI8, TFL_Uint8]>:$lhs,
      TensorOf<[F32, I32, I64, I8, QI8, QUI8, TFL_Uint8]>:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];

  let hasOptions = 0;
}

def TFL_LocalResponseNormalizationOp : TFL_Op<"local_response_normalization",
                                             [NoSideEffect]> {
    let summary = "Local Response Normalization.";

  let description = [{
The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
dimension), and each vector is normalized independently.  Within a given vector,
each component is divided by the weighted, squared sum of inputs within
`depth_radius`.  In detail,

    sqr_sum[a, b, c, d] =
        sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    output = input / (bias + alpha * sqr_sum) ** beta

For details, see [Krizhevsky et al., ImageNet classification with deep
convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
  }];

  let arguments = (ins
      TensorOf<[F32, QI8, QUI8]>:$input,
      I32Attr:$radius,
      F32Attr:$bias,
      F32Attr:$alpha,
      F32Attr:$beta
  );

  let results = (outs
    TensorOf<[F32, QI8, QUI8]>:$output
  );

  let hasOptions = 1;
}

def TFL_GreaterEqualOp : TFL_Op<"greater_equal", [
    ResultsBroadcastableShape, NoSideEffect, NoQuantizableResult]> {
  let summary = "Greater_equal operator";

  let description = [{
    Element-wise greater_equal operation.
  }];

  let arguments = (
      ins AnyTensor:$lhs,
      AnyTensor:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];

  let hasOptions = 0;
}

def TFL_MatrixDiagOp : TFL_Op<"matrix_diag", [
  NoSideEffect,
  TFL_OperandHasAtleastRank<0, 1>,
  PredOpTrait<"operand and result must have the same element type",
    TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = [{
    Returns a tensor with the provided diagonal and everything else padded with zeros.
  }];

  let description = [{
    Given a diagonal, returns a tensor with the diagonal and everything else padded with zeros.
    Assume diagonal has k dimensions `[I, J, K, ..., N]`, then the output is a tensor of rank `k+1`
    with dimensions `[I, J, K, ..., N, N]` where:
       `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n].`
  }];

  let arguments = (ins
    TensorOf<[F32, I8, I64, I32, TFL_Uint8]>:$diagonal
  );

  let results = (outs
    TensorOf<[F32, I8, I64, I32, TFL_Uint8]>:$output
  );

  let hasOptions = 0;
}

// These ops are named NonMaxSuppressionV4 & NonMaxSuppressionV5 to be
// consistent with TensorFlow's naming. They are NOT 'versions' of NMS in the
// sense that one is an incremental change over the other.
// In reality NonMaxSuppressionV5 implements Soft Non Max Suppression and
// NonMaxSuppressionV4 performs hard NMS.

def TFL_NonMaxSuppressionV4Op : TFL_Op<"non_max_suppression_v4", [
  NoSideEffect,
  // Operand 0 (boxes) should have rank 2 with the dim[1] == 4 (box corners)
  TFL_OperandHasRank<0, 2>,
  PredOpTrait<"boxes should have dim[1] == 4",
      TFL_OperandDimEquals<0, 1, 4>>,
  // Operand 1 (scores) should be a 1-dim tensor
  TFL_OperandHasRank<1, 1>,
  // Other operands are scalar params.
  TFL_OperandHasRank<2, 0>, TFL_OperandHasRank<3, 0>,
  TFL_OperandHasRank<4, 0>]> {
  let summary = [{
Greedily selects a subset of bounding boxes in descending order of score,
  }];

  let description = [{
pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes with score less than
`score_threshold` are removed.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system and more
generally is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
  selected_indices = tf.image.non_max_suppression_v2(
      boxes, scores, max_output_size, iou_threshold, score_threshold)
  selected_boxes = tf.gather(boxes, selected_indices)
  }];

  let arguments = (ins
    TFL_FpTensor:$boxes,
    TFL_FpTensor:$scores,
    I32Tensor:$max_output_size,
    TFL_FpTensor:$iou_threshold,
    TFL_FpTensor:$score_threshold
  );

  let results = (outs
    I32Tensor:$selected_indices,
    I32Tensor:$valid_outputs
  );
}

def TFL_NonMaxSuppressionV5Op : TFL_Op<"non_max_suppression_v5", [
  NoSideEffect,
  // Operand 0 (boxes) should have rank 2 with the dim[1] == 4 (box corners)
  TFL_OperandHasRank<0, 2>,
  PredOpTrait<"boxes should have dim[1] == 4",
      TFL_OperandDimEquals<0, 1, 4>>,
  // Operand 1 (scores) should be a 1-dim tensor
  TFL_OperandHasRank<1, 1>,
  // Other operands are scalar params.
  TFL_OperandHasRank<2, 0>, TFL_OperandHasRank<3, 0>,
  TFL_OperandHasRank<4, 0>, TFL_OperandHasRank<5, 0>]> {
  let summary = [{
Greedily selects a subset of bounding boxes in descending order of score,
  }];

  let description = [{
pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes with score less than
`score_threshold` are removed.  Bounding boxes are supplied as
[y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system and more
generally is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
  selected_indices = tf.image.non_max_suppression_v2(
      boxes, scores, max_output_size, iou_threshold, score_threshold)
  selected_boxes = tf.gather(boxes, selected_indices)
This op also supports a Soft-NMS (with Gaussian weighting) mode (c.f.
Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
of other overlapping boxes instead of directly causing them to be pruned.
To enable this Soft-NMS mode, set the `soft_nms_sigma` parameter to be
larger than 0.
  }];

  let arguments = (ins
    TFL_FpTensor:$boxes,
    TFL_FpTensor:$scores,
    I32Tensor:$max_output_size,
    TFL_FpTensor:$iou_threshold,
    TFL_FpTensor:$score_threshold,
    TFL_FpTensor:$soft_nms_sigma
  );

  let results = (outs
    I32Tensor:$selected_indices,
    TFL_FpTensor:$selected_scores,
    I32Tensor:$valid_outputs
  );
}

def TFL_NotEqualOp : TFL_Op<"not_equal", [
    ResultsBroadcastableShape, Commutative, NoSideEffect, NoQuantizableResult]> {
  let summary = "Not_equal operator";

  let description = [{
    Element-wise not_equal operation.
  }];

  let arguments = (
      ins AnyTensor:$lhs,
      AnyTensor:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let builders =
  [
    OpBuilder<
      "Builder *builder, OperationState &result, Value lhs, Value rhs",
      [{
        buildComparisonBinOp(builder, result, lhs, rhs);
      }]>
  ];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def TFL_DivOp : TFL_Op<"div", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Division operator";

  let description = [{
    Element-wise division operation.
  }];

  let arguments = (
      ins AnyTensor:$lhs,
      AnyTensor:$rhs,
      TFL_AFAttr:$fused_activation_function);

  let results = (outs AnyTensor:$output);

  let builders = [TFL_FusedBroadcastableBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];

  let hasOptions = 1;

  let hasFolder = 1;
}

def TFL_EluOp: TFL_Op<"elu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Exponential Linear Unit operator";
  let description = [{
    Computes the exponential linear
      f(x) -> exp(x) - 1 for x < 0, x for x >= 0.
    element-wise.
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs AnyTensor:$y);

  let hasOptions = 0;
}

def TFL_EmbeddingLookupOp: TFL_Op<"embedding_lookup",
    [NoSideEffect,
     PredOpTrait<"value and output must have same element type",
       TCresVTEtIsSameAsOp<0, 1>>
    ]> {
  let summary = "Embedding lookup operator";

  let description = [{
    Looks up ids in a list of embedding tensors.
  }];

  let arguments = (ins
    TensorOf<[I32]>:$lookup,
    TensorOf<[F32, I8, TFL_Uint8]>:$value
   );

  let results = (outs TensorOf<[F32, I8, TFL_Uint8]>:$output);
}

def TFL_EqualOp: TFL_Op<"equal", [Commutative, ResultsBroadcastableShape,
    NoQuantizableResult,
    PredOpTrait<"Operands have same value type", TCopVTEtIsSameAs<0, 1>>]> {
  let summary = "Equal operator";

  let description = [{
    Returns the truth element of x == y element-wise
  }];

  let arguments = (
    ins
    TensorOf<[I1, F32, I32, I64, I8, QI8, QUI8, TFL_Uint8]>:$x,
    TensorOf<[I1, F32, I32, I64, I8, QI8, QUI8, TFL_Uint8]>:$y
  );

  let results = (outs TFL_BoolTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];
}

def TFL_ExpOp: TFL_Op<"exp", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Natural exponentiation operator";

  let description = [{
    Performs element-wise natural exponentiation operation on input.
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasOptions = 0b1;
}

def TFL_ExpandDimsOp: TFL_Op<"expand_dims", [
    NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Inserts a dimension of 1 into a tensor's shape.";

  let description = [{
Given a tensor `input`, this operation inserts a dimension of 1 at the
dimension index `axis` of `input`'s shape. The dimension index `axis` starts at
zero; if you specify a negative number for `axis` it is counted backward from
the end.

This operation is useful if you want to add a batch dimension to a single
element. For example, if you have a single image of shape `[height, width,
channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
which will make the shape `[1, height, width, channels]`.

Other examples:

```
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```

This operation requires that:

`-1-input.dims() <= dim <= input.dims()`

This operation is related to `squeeze()`, which removes dimensions of
size 1.
  }];

  // TODO: Restriction on dim's size and valid range are not modeled here.
  let arguments = (ins AnyTensor:$input, TFL_IntTensor:$dim);

  let results = (outs AnyTensor:$output);

  let hasOptions = 1;
}

def TFL_SqueezeOp: TFL_Op<"squeeze", [NoSideEffect,
                                      SameOperandsAndResultsScale]> {
  let summary = "Removes dimensions of size 1 from the shape of a tensor.";

  let description = [{
Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`axis`.

For example:

```
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
```

Or, to remove specific size 1 dimensions:

```
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
```
  }];

  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$squeeze_dims
  );

  let results = (outs
    AnyTensor:$output
  );

  let hasOptions = 1;

  let customOption = "SqueezeOptions";
}

def TFL_FillOp: TFL_Op<"fill", [NoSideEffect]> {
  let summary = "Fill the tensor with given value.";
  let description = [{
    Fill the tensor with given value.
  }];

  let arguments = (ins TFL_I32OrI64Tensor:$dims,
                  AnyTensor:$value);

  let results = (outs AnyTensor:$res);

  let hasOptions = 0;
}

def TFL_FloorOp: TFL_Op<"floor", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Floor operator";

  let description = [{
    Returns element-wise floor value of the input.
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);
}

def TFL_FloorDivOp : TFL_Op<"floor_div", [
    ResultsBroadcastableShape, NoSideEffect, BinaryOpSameElementTypeConstraint]> {
  let summary = "Floor div operator";

  let description = [{
    Element-wise floor div operation.
  }];

  let arguments = (
    ins AnyTensor:$lhs, AnyTensor:$rhs);

  let results = (outs AnyTensor:$output);

  let builders = [TFL_BroadcastableBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def TFL_FloorModOp : TFL_Op<"floor_mod", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Division reminder";

  let description = [{
    Element-wise division reminder operation.
  }];

  let arguments = (
    ins TensorOf<[I32, I64, F32]>:$lhs,
    TensorOf<[I32, I64, F32]>:$rhs);

  let results = (outs TensorOf<[I32, I64, F32]>:$output);

  let builders = [TFL_BroadcastableBinaryBuilder];
}

def TFL_GreaterOp : TFL_Op<"greater", [
    ResultsBroadcastableShape, NoSideEffect, NoQuantizableResult]> {
  let summary = "Greater operator";

  let description = [{
    Element-wise greater operation.
  }];

  let arguments = (
    ins AnyTensor:$lhs,
    AnyTensor:$rhs);

  let results = (outs AnyTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def TFL_HardSwishOp: TFL_Op<"hard_swish", [NoSideEffect,
                                          SameOperandsAndResultType]> {
  let summary = "Hardswish activation function.";
  let description = [{
    Computes hard-swish activation function
      f(x) -> (x * relu6(x+3))/6
    element-wise.
  }];

  let arguments = (ins TensorOf<[F32, QUI8, QI8]>:$input);

  let results = (outs TensorOf<[F32, QUI8, QI8]>:$out);

  let hasOptions = 0;
}

def TFL_L2NormalizationOp : TFL_Op<"l2_normalization", [NoSideEffect,
    // central_value = min_value / 2 + (max_value - 1) / 2 + 1
    // zero_point = central_value
    // scale = 1. / (central_value - min_value)
    FixedResultScale<Int8UniformQuantizedType<0, 78125, -7>>,
    FixedResultScale<UInt8UniformQuantizedType<128, 78125, -7>>]> {
  let summary = "L2 Normalize Operator";

  let description = [{
    L2Normalization Op
  }];

  let arguments = (ins
    TensorOf<[F32, QUI8, QI8, QUI16, QI16, I8]>:$input,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs TensorOf<[F32, QUI8, QI8, QUI16, QI16, I8]>:$output);

  let hasOptions = 1;

  let customOption = "L2NormOptions";
}

def TFL_LeakyReluOp: TFL_Op<"leaky_relu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Leaky Relu operator";

  // TODO(jpienaar): Add type restriction. This op is only defined for
  // restricted (floating point) types.
  let description = [{
    Element-wise Leaky ReLU operator
      x -> x >= 0 ? x : (alpha * x)
  }];

  let arguments = (
    ins AnyTensor:$input,
    // Slope of the activation function at x < 0.
    F32Attr:$alpha
  );

  let results = (outs AnyTensor:$output);

  let hasOptions = 0b1;
}

def TFL_LessOp : TFL_Op<"less", [
    ResultsBroadcastableShape, NoSideEffect, NoQuantizableResult]> {
  let summary = "Less operator";

  let description = [{
    Element-wise less operation.
  }];

  let arguments = (
    ins AnyTensor:$lhs,
    AnyTensor:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def TFL_LogicalAndOp : TFL_Op<"logical_and", [NoSideEffect]> {
  let summary = "Logical AND operator";

  let description = [{
    Element-wise logical AND operation.
  }];

  let arguments = (
    ins I1Tensor:$lhs,
    I1Tensor:$rhs);

  let results = (outs I1Tensor:$output);

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def TFL_LogicalNotOp : TFL_Op<"logical_not", [NoSideEffect, NoQuantizableResult]> {
  let summary = "Logical NOT operator";

  let description = [{
    Element-wise logical NOT operation.
  }];

  let arguments = (ins I1Tensor:$lhs);

  let results = (outs I1Tensor:$output);
}

def TFL_LogicalOrOp : TFL_Op<"logical_or", [NoSideEffect]> {
  let summary = "Logical OR operator";

  let description = [{
    Element-wise logical OR operation.
  }];

  let arguments = (
    ins I1Tensor:$lhs,
    I1Tensor:$rhs);

  let results = (outs I1Tensor:$output);

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def TFL_LogisticOp: TFL_Op<"logistic", [
    NoSideEffect,
    SameOperandsAndResultShape,
    // zero_point = 0
    // scale = 1. / (max_value + 1)
    FixedResultScale<Int8UniformQuantizedType<-128, 390625, -8>>,
    FixedResultScale<UInt8UniformQuantizedType<0, 390625, -8>>]> {
  let summary = "Logistic operator";

  let description = [{
    Computes element-wise Sigmoid of input
  }];

  let arguments = (ins TensorOf<[AnyFloat, QI8, QUI8, QI16, QUI16]>:$x);

  let results = (outs TensorOf<[AnyFloat, QI8, QUI8, QI16, QUI16]>:$y);
}

def TFL_LogOp: TFL_Op<"log", [
    NoSideEffect, SameOperandsAndResultType, NoQuantizableResult]> {
  let summary = "Natural logarithm operator";

  let description = [{
    Performs element-wise natural logarithm operation on input.
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasFolder = 1;
}

// TODO(b/130643170): Adds some constraint for the input/output element types.
def TFL_LogSoftmaxOp : TFL_Op<"log_softmax", [
    NoSideEffect,
    SameOperandsAndResultShape,
    // zero_point = max_value
    // scale = -log_softmax_output_min / (max_value + 1)
    FixedResultScale<Int8UniformQuantizedType<127, 625, -4>>,
    FixedResultScale<UInt8UniformQuantizedType<255, 625, -4>>]> {
  let summary = "Log softmax operator";

  let description = [{
    Computes element-wise log softmax activations with the following formula

      input - log(reduce_sum(exp(input), dim))
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs AnyTensor:$output);

  let hasOptions = 1;
}

// TODO(ashwinm): Revisit the granularity of the PredOpTraits. We could
// break this into smaller PredOpTraits, each with more descriptive messages
// that would make it easier to trace failures OR, need a way to specify desc
// per Predicate inside the trait and get tablegen to use that to emit error
// message.
def MaxPoolOperandAndResultConstraints : PredOpTrait<"MaxPool2D operand and "
    "result types match specified constraints",
  And<[
    // The input and output tensors should have the same elemental type
    // and they should be one of the specified types below.
    TCopVTEtIs<0, AnyTypeOf<[F32, QI8, QUI8]>>,
    TFL_TCresVTEtIsSameAsOp<0, 0>]>>;

def TFL_MaxPool2DOp : TFL_Op<"max_pool_2d", [
    NoSideEffect,
    MaxPoolOperandAndResultConstraints,
    SameOperandsAndResultsScale]> {
  let summary = "Max Pool 2D op";

  let description = [{
    Performs max pool 2D on input.

    Inputs:
      `inputs[0]`: required: the input tensor
  }];

  let arguments = (
    ins AnyTensor:$input,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_w,
    I32Attr:$stride_h,
    I32Attr:$filter_width,
    I32Attr:$filter_height,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs AnyTensor:$output);

  let hasOptions = 1;

  let customOption = "Pool2DOptions";
}

def TFL_MaxPoolingWithArgMax2DOp :
    Op<TFL_Dialect, "max_pooling_with_argmax_2d", [NoSideEffect]> {
  let summary = "Max Pool 2D with argmax op";

  let description = [{
    Performs max pooling on the input and outputs both max values and indices.
    Each index is a flatten index in a sub-array of "filter_w" x "filter_h" size
    Note this is a custom op that is not supported in the standard runtime.

    Inputs:
      `inputs[0]`: required: the input activation tensor
  }];

  let arguments = (
    ins AnyTensor:$input,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_w,
    I32Attr:$stride_h,
    I32Attr:$filter_w,
    I32Attr:$filter_h
  );

  let results = (outs
    AnyTensor:$value,
    AnyTensor:$indices
  );
}

def TFL_MaxUnpooling2DOp :
    Op<TFL_Dialect, "max_unpooling_2d", [NoSideEffect]> {
  let summary = "Max Unpool 2D";

  let description = [{
    Performs max unpool operation.
    To some extent this is the reverse operation of max pooling:
    the elements in the input activation tensor is stored into the position
    specified by the input indices.
    Note this is a custom op that is not supported in the standard runtime.

    Inputs:
      `inputs[0]`: required: the input activation tensor
      `inputs[1]`: required: the input indices
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$indices,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_w,
    I32Attr:$stride_h,
    I32Attr:$filter_w,
    I32Attr:$filter_h
  );

  let results = (outs AnyTensor:$outputs);
}

def TFL_MaximumOp : TFL_Op<"maximum", [
    ResultsBroadcastableShape, NoSideEffect, Commutative, SameOperandsAndResultsScale,
    TFL_OperandHasRankLessThan<0, 4>, TFL_OperandHasRankLessThan<1, 4>]> {
  let summary = "Max operator";
  let description = [{
    Element-wise max operation.
  }];

  let arguments = (
    ins TensorOf<[AnyFloat, TFL_Int32Or64, QI8, QUI8]>:$lhs,
    TensorOf<[AnyFloat, TFL_Int32Or64, QI8, QUI8]>:$rhs
  );

  let results = (outs
    TensorOf<[AnyFloat, TFL_Int32Or64, QI8, QUI8]>:$max
  );

  let builders = [TFL_BroadcastableBinaryBuilder];

  let hasOptions = 0;
}

def TFL_MeanOp : TFL_Op<"mean", [NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Mean operator";

  let description = [{
    Computes the mean of elements across dimensions of a tensor.
    Reduces input_tensor along the dimensions given in axis.
    Unless keepdims is true, the rank of the tensor is reduced by 1 for
    each entry in axis. If keepdims is true, the reduced dimensions are retained
    with length 1.
  }];

  let arguments = (ins
    TensorOf<[F32, I8, I32, I64, QI8, QUI8, TFL_Uint8]>:$input,
    TensorOf<[I32, I64]>:$axis,
    BoolAttr:$keep_dims
  );

  let results = (outs
    TensorOf<[F32, I32, I64, I8, QI8, QUI8, TFL_Uint8]>:$output);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_OneHotOp : TFL_Op<"one_hot", [NoSideEffect]> {
  let summary = "OneHot operator";

  let description = [{
    Returns a one-hot tensor.The locations represented by indices in `indices`
    take value `on_value`, while all other locations take value `off_value`.

    If the input `indices` is rank `N`, the output will have rank `N+1`,
    The new axis is created at dimension `axis` (default: the new axis is
    appended at the end).
  }];

  let arguments = (ins
    TensorOf<[I32, I64]>:$indices,
    I32Tensor:$depth,
    TensorOf<[F32, I32, I64, I1]>:$on_value,
    TensorOf<[F32, I32, I64, I1]>:$off_value,

    I32Attr:$axis
  );

  let results = (outs
    TensorOf<[F32, I32, I64, I1]>:$output
  );

  let hasOptions = 1;
}

def TFL_RoundOp: TFL_Op<"round", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Round operator";

  let description = [{
Rounds the values of a tensor to the nearest integer, element-wise.
  }];

  let arguments = (ins
    TensorOf<[F32]>:$x
  );

  let results = (outs
    TensorOf<[F32]>:$y
  );
}

def TFL_SliceOp : TFL_Op<"slice", [
    NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Return a slice from 'input'.";

  let description = [{
The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.

`begin` is zero-based; `size` is one-based. If size[i] is -1, all remaining
elements in dimension i are included in the slice. In other words, this is
equivalent to setting:
  size[i] = input.dim_size(i) - begin[i]

*Requirements*:
  0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
  }];

  let arguments = (ins
    AnyTensor:$input,
    TFL_I32OrI64Tensor:$begin,
    TFL_I32OrI64Tensor:$size
  );

  let results = (outs
    AnyTensor:$output
  );

  let verifier = [{ return Verify(*this); }];
}

def TFL_SumOp: TFL_Op<"sum", [NoSideEffect]> {
  let summary = "Sum operator";

  let description = [{
    Computes the sum reduction along the specified axes
  }];

  let arguments = (ins
    AnyTensor:$input,
    I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs AnyTensor);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_ReduceMinOp: TFL_Op<"reduce_min", [
    NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Min-reduction operator";

  let description = [{
    Computes the min reduction along the specified axes
  }];

  let arguments = (ins
    AnyTensor:$input,
    I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs AnyTensor);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_ReduceMaxOp: TFL_Op<"reduce_max", [
    NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Max-reduction operator";

  let description = [{
    Computes the max reduction along the specified axes
  }];

  let arguments = (ins
    AnyTensor:$input,
    I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs AnyTensor);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_ReduceProdOp: TFL_Op<"reduce_prod", [NoSideEffect]> {
  let summary = "Prod-reduction operator";

  let description = [{
    Computes the product along the specified axes
  }];

  let arguments = (ins
    TensorOf<[F32, I8, I32, I64]>:$input,
    I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs AnyTensor);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_MinimumOp : TFL_Op<"minimum", [
    ResultsBroadcastableShape, NoSideEffect, Commutative, SameOperandsAndResultsScale,
    TFL_OperandHasRankLessThan<0, 4>, TFL_OperandHasRankLessThan<1, 4>]> {
  let summary = "Min operator";
  let description = [{
    Element-wise min operation.
  }];

  let arguments = (
    ins TensorOf<[AnyFloat, TFL_Int32Or64, QI8, QUI8]>:$lhs,
    TensorOf<[AnyFloat, TFL_Int32Or64, QI8, QUI8]>:$rhs
  );

  let results = (outs
    TensorOf<[AnyFloat, TFL_Int32Or64, QI8, QUI8]>:$min
  );

  let builders = [TFL_BroadcastableBinaryBuilder];

  let hasOptions = 0;
}

def TFL_MulOp : TFL_Op<"mul", [ResultsBroadcastableShape, NoSideEffect, Commutative]> {
  let summary = "Multiplication operator";

  let description = [{
    Element-wise multiplication operation.
  }];

  let arguments = (
    ins AnyTensor:$lhs,
    AnyTensor:$rhs,
    TFL_AFAttr:$fused_activation_function);

  let results = (outs AnyTensor:$output);

  let hasFolder = 1;

  let builders = [TFL_FusedBroadcastableBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];

  let hasOptions = 1;
}

def TFL_NegOp: TFL_Op<"neg", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Negation operator";

  let description = [{
    Computes element-wise negation of input
  }];

  let arguments = (ins AnyTensor:$x);

  let results = (outs AnyTensor:$y);

  let hasOptions = 0b1;

  let hasFolder = 1;
}

def TFL_PackOp : TFL_Op<"pack", [NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Packs a list of tensors along a dimension into one tensor";

  let description = [{
    Packs a list of `values_count` rank-`R` tensors into one rank-`(R+1)`
    tensor.

    Packs the `values_count` tensors in `values` into a tensor with rank one
    higher than each tensor in `values`, by packing them along the `axis`
    dimension.

    Given a list of tensors of shape `(A, B, C)`;

    if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
    if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
    Etc.

    For example:

    ```
    # 'x' is [1, 4]
    # 'y' is [2, 5]
    # 'z' is [3, 6]
    pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
    pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
    ```

    This is the opposite of `unpack`.
  }];

  let arguments = (ins
    Variadic<TensorOf<[F32, I8, I16, I32, I64, QI8, QUI8, QI16]>>:$values,

    I32Attr:$values_count,
    I32Attr:$axis
  );

  let results = (outs
    TensorOf<[F32, I8, I16, I32, I64, QI8, QUI8, QI16]>:$output
  );

  let verifier = [{ return Verify(*this); }];

  let hasCanonicalizer = 1;

  let hasOptions = 1;
}

def TFL_PadOp : TFL_Op<"pad", [
    NoSideEffect,
    SameOperandsAndResultsScale,
    TFL_OperandHasRank<1, 2>,
    TFL_OperandRankEquals1DimOfOperand<0, 1>]> {
  let summary = "Padding operator";

  let description = [{
    This operation pads a `input` with zeros according to the `paddings` you
    specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
    the rank of `input`. For each dimension D of `input`, `paddings[D, 0]`
    indicates how many zeros to add before the contents of `input` in that
    dimension, and `paddings[D, 1]` indicates how many zeros to add after the
    contents of `input` in that dimension.

    The padded size of each dimension D of the output is:

      `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

    For example:

    ```
    # 't' is [[1, 1], [2, 2]]
    # 'paddings' is [[1, 1], [2, 2]]
    # rank of 't' is 2
    pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                          [0, 0, 1, 1, 0, 0]
                          [0, 0, 2, 2, 0, 0]
                          [0, 0, 0, 0, 0, 0]]
    ```
  }];

  let arguments = (
    ins TensorOf<[F32, I8, I32, I64, QI8, QUI8]>:$input,
    TFL_I32OrI64Tensor:$padding);

  let results = (outs TensorOf<[F32, I8, I32, I64, QI8, QUI8]>:$output);

  let hasOptions = 1;
}

def TFL_PadV2Op : TFL_Op<"padv2", [
    NoSideEffect,
    SameOperandsAndResultsScale,
    TFL_OperandHasRank<1, 2>,
    TFL_OperandHasRank<2, 0>,
    TFL_OperandRankEquals1DimOfOperand<0, 1>,
    PredOpTrait<"input and constant value operands must have same element type",
      TCopVTEtAreSameAt<[0, 2]>>]> {
  let summary = "Padding operator v2";

  let description = [{
    This operation pads a `input` according to the `paddings` and
    `constant_values` you specify. `paddings` is an integer tensor with shape
    `[Dn, 2]`, where n is the rank of `input`. For each dimension D of `input`,
    `paddings[D, 0]` indicates how many zeros to add before the contents of
    `input` in that dimension, and `paddings[D, 1]` indicates how many zeros to
    add after the contents of `input` in that dimension. `constant_values` is a
    scalar tensor of the same type as `input` that indicates the value to use
    for padding `input`.

    The padded size of each dimension D of the output is:

      `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

    For example:

    ```
    # 't' is [[1, 1], [2, 2]]
    # 'paddings' is [[1, 1], [2, 2]]
    # rank of 't' is 2
    pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                          [0, 0, 1, 1, 0, 0]
                          [0, 0, 2, 2, 0, 0]
                          [0, 0, 0, 0, 0, 0]]
    ```
  }];

  let arguments = (
    ins TensorOf<[F32, I8, I32, I64, QI8, QUI8]>:$input,
    TFL_I32OrI64Tensor:$padding,
    TensorOf<[F32, I8, I32, I64]>:$constant_values);

  let results = (outs TensorOf<[F32, I8, I32, I64, QI8, QUI8]>:$output);

  let hasOptions = 1;
}

def TFL_PowOp : TFL_Op<"pow", [ResultsBroadcastableShape, NoSideEffect, NoQuantizableResult]> {
  let summary = "Power operator";

  let description = [{
    Element-wise power operation.
  }];

  let arguments = (
    ins AnyTensor:$lhs,
    AnyTensor:$rhs);

  let results = (outs AnyTensor:$output);

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];

  let builders = [TFL_BroadcastableBinaryBuilder];
}

def TFL_PReluOp : TFL_Op<"prelu", [NoSideEffect]> {
  let summary = "Parameterized Relu operator";

  let description = [{
    Parameterized Relu operator
      x -> x >= 0 ? x : (alpha * x)
    where alpha is a trainable tensor.
    alpha should have one less rank than the input as it doesn't have the batch
    dimension, and the other dimensions either should be the same size as input
    or size 1, where it is broadcasted in the second case.
  }];

  let arguments = (
    ins TensorOf<[F32, QUI8]>:$input,
    TensorOf<[F32, QUI8]>:$alpha
  );

  let results = (outs TensorOf<[F32, QUI8]>:$output);

  let verifier = [{ return Verify(*this); }];
}

def TFL_RankOp: TFL_Op<"rank", [NoSideEffect]> {
  let summary = "Rank operator.";
  let description = [{
    Returns the rank of a tensor.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs TFL_IntTensor:$output);

  let hasFolder = 1;
}

def TFL_ReluOp: TFL_Op<"relu", [NoSideEffect,
                                SameOperandsAndResultShape,
                                SameOperandsAndResultsScale]> {
  let summary = "Relu operator";

  let description = [{
    Element-wise Relu operator
      x -> max(0, x)
  }];

  let arguments = (ins TensorOf<[F32, QUI8, I8]>:$x);

  let results = (outs TensorOf<[F32, QUI8, I8]>:$y);
}

def TFL_Relu6Op: TFL_Op<"relu6", [NoSideEffect,
                                  SameOperandsAndResultShape,
                                  SameOperandsAndResultsScale]> {
  let summary = "Relu6 operator";

  let description = [{
    Element-wise Relu6 operator
      x -> max(0, min(6, x))
  }];

  let arguments = (ins TensorOf<[F32, QUI8, I8]>:$x);

  let results = (outs TensorOf<[F32, QUI8, I8]>:$y);
}

def TFL_Relu1Op: TFL_Op<"relu_n1_to_1", [NoSideEffect,
                                  SameOperandsAndResultShape,
                                  SameOperandsAndResultsScale]> {
  let summary = "Relu1 operator";

  let description = [{
    Element-wise Relu1 operator
      x -> max(-1, min(1, x))
  }];

  let arguments = (ins TensorOf<[F32, QUI8, I8]>:$x);

  let results = (outs TensorOf<[F32, QUI8, I8]>:$y);
}

def TFL_ReshapeOp: TFL_Op<"reshape", [
    NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Reshape operator";

  let description = [{
    Produces a tensor with the same values but different static shape defined
    by the output type.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Tensor:$shape);

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 0b1;
  let hasFolder = 1;
}

def TFL_ReverseSequenceOp : TFL_Op<"reverse_sequence", [NoSideEffect]> {
  let summary = "Reverses variable length slices.";

  let description = [{
This op first slices `input` along the dimension `batch_dim`, and for each
slice `i`, reverses the first `seq_lengths[i]` elements along
the dimension `seq_dim`.

The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

The output slice `i` along dimension `batch_dim` is then given by input
slice `i`, with the first `seq_lengths[i]` slices along dimension
`seq_dim` reversed.
  }];

  let arguments = (ins
    TensorOf<[F32, I16, I32, I64, TFL_Uint8]>:$input,
    TFL_I32OrI64Tensor:$seq_lengths,

    I32Attr:$seq_dim,
    I32Attr:$batch_dim
  );

  let results = (outs
    TensorOf<[F32, I16, I32, I64, TFL_Uint8]>:$output
  );

  let hasOptions = 1;
}

def TFL_RsqrtOp: TFL_Op<"rsqrt", [NoSideEffect, SameOperandsAndResultType, NoQuantizableResult]> {
  let summary = "Reciprocal of square root operator";

  let description = [{
    Computes element-wise reverse square root of input
  }];

  let arguments = (ins AnyTensor:$x);

  let results = (outs AnyTensor:$y);

  let hasFolder = 1;
}

def TFL_ShapeOp: TFL_Op<"shape", [NoSideEffect]> {
  let summary = "Shape operator";

  let description = [{
    Returns the shape of a tensor.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs AnyTensor:$output);

  DerivedTypeAttr out_type = DerivedTypeAttr<[{
    return getResult().getType().cast<TensorType>().getElementType();
  }]>;

  let hasOptions = 1;
}

// TODO(jpienaar): Flesh this out.
def TFL_RangeOp: TFL_Op<"range", [NoSideEffect, TFL_OperandHasRank<0, 0>,
    TFL_OperandHasRank<1, 0>, TFL_OperandHasRank<2, 0>,
    PredOpTrait<"operands and output must have same element type",
      And<[TCresVTEtIsSameAsOp<0, 0>, TCresVTEtIsSameAsOp<0, 1>,
           TCresVTEtIsSameAsOp<0, 2>]>>]> {
  let summary = "Range operator";

  let description = [{
    Returns a 1D tensor defined by a sequence from `start` to `limit` with
    a given `delta`.
  }];

  let arguments = (ins
    AnyTensor:$start,
    AnyTensor:$limit,
    AnyTensor:$delta);

  let results = (outs AnyTensor:$result);

  let hasFolder = 1;
}

def TFL_ReverseV2Op: TFL_Op<"reverse_v2",
                            [NoSideEffect, TFL_OperandHasRank<1,1>]> {
  let summary = "ReverseV2 Operator";

  let description = [{
    Reverses specific dimensions of a tensor.

    Given a tensor, and a int32/int64 tensor axis representing the set
    of dimensions of tensor to reverse.
    This operation reverses each dimension i for
    which there exists j s.t. axis[j] == i.

    Args:
      tensor: A Tensor. Must be one of the following types:
      uint8, int16, int32, int64, float32, bool Up to 8-D.

      axis: A Tensor. Must be one of the following types: int32, int64.
      with only 1 element which is the axis index.
      TODO: Add support for multiple elements.
  }];

  let arguments = (
    ins
    TensorOf<[F32, I16, I32, I64, TFL_Uint8, I1]>:$input,
    TensorOf<[I32, I64]>:$axis
  );

  let results = (outs
  TensorOf<[F32, I16, I32, I64, TFL_Uint8, I1]>:$output
  );
}

// Select has many instances in TF models where one or more of its operands
// are unranked. Therefore, we skip adding shape constraints here.
def TFL_SelectOp : TFL_Op<"select", [NoSideEffect,
  PredOpTrait<"operands have same element type", TCopVTEtIsSameAs<1, 2>>,
  PredOpTrait<"operands and result have same element type",
    TCresVTEtIsSameAsOp<0, 1>>]> {
  let summary = "Select operator";

  let description = [{
    Select values of 'x' if the corresponding value of 'condition' is true or
    the value of 'y' if false. There are valid condition input sizes:

    1. Either the same shape (in which case the select is elementwise), or
    2. condition must be Rank 1 and match over the first dimension.
  }];

  let arguments = (ins
    TFL_BoolTensor:$condition,
    TensorOf<[F32, I1, I8, I16, I32, I64, TFL_Uint8]>:$x,
    TensorOf<[F32, I1, I8, I16, I32, I64, TFL_Uint8]>:$y);
  let results = (outs AnyTensor:$output);

  // TODO(jpienaar): autogenerate this.
  let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                            "Value condition, Value x, Value y",
  [{
    auto resultType = x.getType();
    result.addOperands({condition, x, y});
    result.types.push_back(resultType);
  }]>];

  let hasOptions = 1;
}

def TFL_SelectV2Op : TFL_Op<"select_v2", [NoSideEffect]> {
  let summary = "SelectV2 operator";

  let description = [{
    Select values of 'x' if the corresponding value of 'condition' is true or
    the value of 'y' if false. There are valid condition input sizes:

    1. Either the same shape (in which case the select is elementwise), or
    2. Broadcastable shapes between 'condition', 'x' and 'y'.
  }];

  let arguments = (ins
    TFL_BoolTensor:$condition,
    TensorOf<[F32, I1, I8, I16, I32, I64, TFL_Uint8]>:$x,
    TensorOf<[F32, I1, I8, I16, I32, I64, TFL_Uint8]>:$y);
  let results = (outs AnyTensor:$output);

  let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                            "Value cond, Value x, Value y",
  [{
    BuildSelectV2Op(builder, result, cond, x, y);
  }]>];

  let hasOptions = 1;
}

def TFL_SinOp: TFL_Op<"sin", [
    NoSideEffect, SameOperandsAndResultType, NoQuantizableResult]> {
  let summary = "Sine operator";

  let description = [{
    Computes element-wise Sine of input
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasFolder = 1;
}

// TODO(b/130643170): Adds some constraint for the input/output element types.
def TFL_SoftmaxOp : TFL_Op<"softmax", [
    NoSideEffect,
    SameOperandsAndResultShape,
    // zero_point = 0
    // scale = 1. / (max_value + 1)
    FixedResultScale<Int8UniformQuantizedType<-128, 390625, -8>>,
    FixedResultScale<UInt8UniformQuantizedType<0, 390625, -8>>]> {
  let summary = "Softmax operator";

  let description = [{
    Computes element-wise softmax activations with the following formula

      exp(input) / tf.reduce_sum(exp(input * beta), dim)
  }];

  let arguments = (
    ins AnyTensor:$input,
    F32Attr:$beta
  );

  let results = (outs AnyTensor:$output);

  let hasOptions = 1;
}

def TFL_SqrtOp: TFL_Op<"sqrt", [
    NoSideEffect, SameOperandsAndResultType, NoQuantizableResult]> {
  let summary = "Square root operator";

  let description = [{
    Computes element-wise Square root of input
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasFolder = 1;
}

def TFL_SquareOp: TFL_Op<"square", [
    NoSideEffect, SameOperandsAndResultType, NoQuantizableResult]> {
  let summary = "Square operator";

  let description = [{
    Computes element-wise Square of input
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasOptions = 0b1;

  let hasFolder = 1;
}

def TFL_SubOp : TFL_Op<"sub", [ResultsBroadcastableShape, NoSideEffect]> {
  let summary = "Subtraction operator";

  let description = [{
    Element-wise subtraction operation.
  }];

  let arguments = (
    ins AnyTensor:$lhs,
    AnyTensor:$rhs,
    TFL_AFAttr:$fused_activation_function);

  let results = (outs AnyTensor:$output);

  let hasFolder = 1;

  let builders = [TFL_FusedBroadcastableBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];

  let hasOptions = 1;
}

// TODO(jpienaar): Expand the kernel implementation to support all types besides
// I32 and F32.
def TFL_SquaredDifferenceOp : TFL_Op<"squared_difference", [
    ResultsBroadcastableShape, NoSideEffect, NoQuantizableResult]> {
  let summary = "Squared difference operator";

  let description = [{
    Element-wise squared difference operation.
  }];

  let arguments = (
    ins AnyTensor:$lhs,
    AnyTensor:$rhs);

  let results = (outs AnyTensor:$output);

  let builders = [TFL_BroadcastableBinaryBuilder];

  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];

  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def TFL_TanhOp: TFL_Op<"tanh", [
    NoSideEffect,
    SameOperandsAndResultShape,
    // central_value = min_value / 2 + (max_value - 1) / 2 + 1
    // zero_point = central_value
    // scale = 1. / (central_value - min_value)
    FixedResultScale<Int8UniformQuantizedType<0, 78125, -7>>,
    FixedResultScale<UInt8UniformQuantizedType<128, 78125, -7>>]> {
  let summary = "Hyperbolic tangent operator";

  let description = [{
    Computes element-wise Hyperbolic tangent of input
  }];

  let arguments = (ins TensorOf<[F32, I16, I8, QI8, QUI8, QI16, QUI16, TFL_Uint8]>:$x);

  let results = (outs TensorOf<[F32, I16, I8, QI8, QUI8, QI16, QUI16, TFL_Uint8]>:$y);
}

def TFL_TileOp: TFL_Op<"tile", [NoSideEffect, SameOperandsAndResultsScale,
  PredOpTrait<"resultant element type needs to match first operand type",
              TFL_TCresVTEtIsSameAsOp<0,0>>]> {
  let summary = "Tile operator.";
  let description = [{
    Constructs a tensor by tiling a given tensor.

   This operation creates a new tensor by replicating input
   multiples times. The output tensor's i'th dimension has
   input.dims(i) * multiples[i] elements, and the values of input
   are replicated multiples[i] times along the 'i'th dimension.
   For example, tiling [a b c d] by [2] produces [a b c d a b c d].
  }];

  let arguments = (ins
    TensorOf<[F32, I1, I32, I64, TFL_Uint8, QUI8]>:$input,
    TFL_I32OrI64Tensor:$multiples);

  let results = (outs
    TensorOf<[F32, I1, I32, I64, TFL_Uint8, QUI8]>:$output);

  let hasOptions = 0;
}

// TODO(jpienaar): Maybe make it accept any single element tensor as `k`.
// TODO(jpienaar): Check that input has one or more dimensions.
// TODO(jpienaar): Check that k is less or equal the internal dimension
def TFL_TopKV2Op: TFL_Op<"topk_v2", [NoSideEffect, TFL_OperandHasRank<1,0>,
    PredOpTrait<"result and input element type match",
                TCresVTEtIsSameAsOp<0,0>>, SameOperandsAndResultsScale]> {
  let summary = "TopK operator";

  let description = [{
    Returns the top `k` largest element along each last dimensional slice of
    `input` and the indices of values within the last dimension of the input
    tensor.
  }];

  let arguments = (ins
    TensorOf<[F32, I8, I32, I64, TFL_Uint8, QI8, QUI8]>:$input,
    I32Tensor:$k);

  let results = (outs
    TensorOf<[F32, I8, I32, I64, TFL_Uint8, QI8, QUI8]>:$values,
    I32Tensor:$indices);

  let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                            "Value input, Value k",
                  [{ BuildTopKOp(builder, result, input, k); }]>];

  let hasOptions = 1;
}

def TFL_TransposeOp : TFL_Op<"transpose",
  [NoSideEffect,
   TFL_OperandHasRank<1,1>,
   // TODO(jpienaar): these are only true dynamically, change so that it works
   // with unknowns.
   // TFL_OperandRankEquals1DimOfOperand<0, 1>,
   PredOpTrait<"input and output must have same element type",
   TCresVTEtIsSameAsOp<0, 0>>,
   SameOperandsAndResultsScale]> {
  let summary = "Transpose operator";

  let description = [{
    Returns the Transpose of x
  }];

  let arguments = (
    ins AnyTensor:$x,
    TensorOf<[I32]>:$perm
  );

  let results = (outs
    AnyTensor:$y
  );

  let verifier = [{ return Verify(*this); }];

  let hasFolder = 1;
}

def TFL_UnpackOp : TFL_Op<"unpack", [NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Unpacks a tensor along a dimension into multiple tensors";

  let description = [{
    Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.

    Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
    For example, given a tensor of shape `(A, B, C, D)`;

    If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
      and each tensor in `output` will have shape `(B, C, D)`. (Note that the
      dimension unpacked along is gone, unlike `split`).

    If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
      and each tensor in `output` will have shape `(A, C, D)`.
    Etc.

    This is the opposite of `pack`.
  }];

  let arguments = (ins
    TensorOf<[F32, I1, I8, I32, QI8, QUI8]>:$input,

    I32Attr:$num,
    I32Attr:$axis
  );

  let results = (outs
    Variadic<TensorOf<[F32, I1, I8, I32, QI8, QUI8]>>:$outputs
  );

  let verifier = [{ return Verify(*this); }];

  let hasOptions = 1;
}

def TFL_ZerosLikeOp: TFL_Op<"zeros_like", [NoSideEffect]> {
  let summary = "ZerosLike operator";

  let description = [{
    Returns a tensor of zeros with the same shape and type as the input tensor.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs AnyTensor:$output);

  let hasOptions = 1;
}

def TFL_BatchToSpaceNdOp: TFL_Op<"batch_to_space_nd", [
    NoSideEffect,
    SameOperandsAndResultsScale,
    PredOpTrait<"input and output must have same element type",
      TCresVTEtIsSameAsOp<0, 0>>
  ]> {
  let summary = "BatchToSpaceNd operator";

  let description = [{
    This operation reshapes the "batch" dimension 0 into space dimensions.
  }];

  let arguments = (ins
    TensorOf<[F32, I8, I32, I64, QI8, QUI8]>:$input,
    TensorOf<[I32]>:$block_shape,
    TensorOf<[I32]>:$indices
  );

  let results = (outs
    TensorOf<[F32, I16, I32, I64, QI8, QUI8]>:$output
  );
}

def TFL_SpaceToBatchNdOp: TFL_Op<"space_to_batch_nd", [
    NoSideEffect,
    SameOperandsAndResultsScale,
    PredOpTrait<"input and output must have same element type",
      TCresVTEtIsSameAsOp<0, 0>>
  ]> {
  let summary = "SpaceToBatchNd operator";

  let description = [{
    This operation reshapes space dimensions into the "batch" dimension 0
  }];

  let arguments = (ins
    TensorOf<[F32, I8, I32, I64, QI8, QUI8]>:$input,
    TensorOf<[I32]>:$block_shape,
    TensorOf<[I32]>:$paddings
  );

  let results = (outs
    TensorOf<[F32, I16, I32, I64, QI8, QUI8]>:$output
  );
}

def TFL_SpaceToDepthOp: TFL_Op<"space_to_depth", [
    NoSideEffect,
    SameOperandsAndResultsScale,
    PredOpTrait<"input and output must have same element type",
      TCresVTEtIsSameAsOp<0, 0>>
  ]> {
  let summary = "SpaceToDepth operator";

  let description = [{
    Rearranges blocks of spatial data, into depth. More specifically,
    this op outputs a copy of the input tensor where values from the `height`
    and `width` dimensions are moved to the `depth` dimension.
    `block_size` indicates the input block size.
   }];

  let arguments = (ins
    TensorOf<[F32, I8, I32, I64, TFL_Uint8, QUI8]>:$input,
    I32Attr:$block_size
  );

  let results = (outs
    TensorOf<[F32, I8, I32, I64, TFL_Uint8, QUI8]>:$output
  );

  let hasOptions = 1;
}

def TFL_DepthToSpaceOp: TFL_Op<"depth_to_space", [
    NoSideEffect,
    SameOperandsAndResultsScale,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>
  ]> {
  let summary = "DepthToSpace operator";

  let description = [{
    Rearranges data from depth into blocks of spatial data.
    This is the reverse transformation of SpaceToDepth. More specifically,
    this op outputs a copy of the input tensor where values from the `depth`
    dimension are moved in spatial blocks to the `height` and `width`
    dimensions. The attr `block_size` indicates the input block size and how
    the data is moved.
   }];

  let arguments = (ins
    TensorOf<[F32, I8, I32, I64, TFL_Uint8, TFL_Quint8, QUI8]>:$input,
    I32Attr:$block_size
  );

  let results = (outs
    TensorOf<[F32, I8, I32, I64, TFL_Uint8, TFL_Quint8, QUI8]>:$output
  );

  let hasOptions = 1;
}

def TFL_SplitOp : TFL_Op<"split", [
    NoSideEffect,
    TFL_Operand0DOr1ElementTensor<0>,
    SameOperandsAndResultsScale]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let description = [{
    Splits the `value` tensor along `split_dim` into a number of sub-tensors
    with same shape as the original one, except for `split_dim`. Same as
    tf.Split.
  }];

  let arguments = (ins
    TensorOf<[I32]>:$split_dim,
    TensorOf<[F32, I16, I32, I64, QI8, QUI8]>:$value,
    PositiveI32Attr:$num_splits
  );

  let results = (outs
    Variadic<TensorOf<[F32, I16, I32, I64, QI8, QUI8]>>:$outputs
  );

  let verifier = [{ return Verify(*this); }];

  let hasOptions = 1;
}

def TFL_SplitVOp : TFL_Op<"split_v", [NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let description = [{
    Splits the `value` tensor along `split_dim` into a number of sub-tensors
    with same shape as the original one, except for `split_dim`. The grouping
    of the resultant sub-tensors is decided by `size-splits`. Same as tf.SplitV.
  }];

  let arguments = (ins
    TensorOf<[F32, I16, I32, I64, QI8, QUI8]>:$value,
    1DTensorOf<[I32]>:$size_splits,
    0DTensorOf<[I32]>:$split_dim,
    PositiveI32Attr:$num_splits
  );

  let results = (outs
    Variadic<TensorOf<[F32, I16, I32, I64, QI8, QUI8]>>:$outputs
  );

  let verifier = [{ return Verify(*this); }];

  let hasOptions = 1;
}

def TFL_ResizeBilinearOp: TFL_Op<"resize_bilinear", [
    NoSideEffect, SameOperandsAndResultsScale]> {
  let summary = "ResizeBilinear Op";

  let description = [{
    Resize `images` to `size` using bilinear interpolation.
  }];

  let arguments = (ins
    // TODO(ycling): Support quantized types.
    TensorOf<[F32, I32, QI8, QUI8]>:$input,
    TensorOf<[I32]>:$size,
    BoolAttr:$align_corners,
    DefaultValuedAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    TensorOf<[F32, QI8, QUI8]>:$output
  );

  let hasOptions = 1;
}

def TFL_ResizeNearestNeighborOp : TFL_Op<"resize_nearest_neighbor",
                                [NoSideEffect,
                                 SameOperandsAndResultsScale]> {
  let summary = "ResizeNearestNeighbor Op";

  let description = [{
    Resize `images` to `size` using nearest neighbor interpolation.
  }];

  let arguments = (ins
    TensorOf<[F32, I8, TFL_Uint8, QUI8, QI8]>:$input,
    TensorOf<[I32]>:$size,
    BoolAttr:$align_corners
  );

  let results = (outs
    TensorOf<[F32, I8, TFL_Uint8, QUI8, QI8]>:$output
  );

  let hasOptions = 1;
}

def TFL_SparseToDenseOp : TFL_Op<"sparse_to_dense", [NoSideEffect]> {
  let summary = "Converts a sparse representation into a dense tensor.";

  let description = [{
Builds an array `dense` with shape `output_shape` such that

```
# If sparse_indices is scalar
dense[i] = (i == sparse_indices ? sparse_values : default_value)

# If sparse_indices is a vector, then for each i
dense[sparse_indices[i]] = sparse_values[i]

# If sparse_indices is an n by d matrix, then for each i in [0, n)
dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
```

All other values in `dense` are set to `default_value`.  If `sparse_values` is a
scalar, all sparse indices are set to this single value.

Indices should be sorted in lexicographic order, and indices must not
contain any repeats. If `validate_indices` is true, these properties
are checked during execution.
  }];

  let arguments = (ins
    TFL_I32OrI64Tensor:$sparse_indices,
    TFL_I32OrI64Tensor:$output_shape,
    TensorOf<[I32, I64, I8, TFL_Uint8, F32]>:$sparse_values,
    TensorOf<[I32, I64, I8, TFL_Uint8, F32]>:$default_value
  );

  let results = (outs
    TensorOf<[I32, I64, I8, TFL_Uint8, F32]>:$dense
  );
}

def TFL_StridedSliceOp: TFL_Op<"strided_slice",
  [
    NoSideEffect,
    PredOpTrait<"input and output must have same element type",
    TFL_TCresVTEtIsSameAsOp<0, 0>>,
    SameOperandsAndResultsScale
  ]> {
  let summary = "StridedSlice Op";

  let description = [{
    Return a strided slice from `input`.
  }];

  let arguments = (ins
    TensorOf<[F32, I32, I64, I8, QI8, QUI8, I1, TFL_Quint8, TFL_Uint8]>:$input,
    TensorOf<[I32]>:$begin,
    TensorOf<[I32]>:$end,
    TensorOf<[I32]>:$strides,

    I32Attr:$begin_mask,
    I32Attr:$end_mask,
    I32Attr:$ellipsis_mask,
    I32Attr:$new_axis_mask,
    I32Attr:$shrink_axis_mask
  );

  let results = (outs
    TensorOf<[F32, I32, I64, I8, QI8, QUI8, I1, TFL_Quint8, TFL_Uint8]>:$output
  );

  let hasOptions = 1;
}

def TFL_CastOp : TFL_Op<"cast", [
    NoSideEffect, SameOperandsAndResultShape, NoQuantizableResult]> {
  let summary = "Cast operator";

  let description = [{
    Casts input from input type to output type.
  }];

  let arguments = (ins
    TensorOf<[F32, I1, I32, I64, TFL_Quint8, TFL_Uint8, Complex<F<32>>]>:$input
  );

  let results = (outs TensorOf<[F32, I1, I32, I64, Complex<F<32>>]>:$output);

  // TFLite's cast op does not utilize CastOptions, instead derives types
  // from the TfLiteTensors.
  let hasOptions = 0;
}


def TFL_MirrorPadOp: TFL_Op<"mirror_pad", [
                     NoSideEffect, TFL_OperandHasRank<1, 2>]> {
  let summary = "MirrorPad Operator. Pads a tensor with mirrored values.";

  let description = [{
    This operation pads a input with mirrored values according to the paddings
    you specify. paddings is an integer tensor with shape [n, 2],
    where n is the rank of input.
    For each dimension D of input, paddings[D, 0] indicates how many values
    to add before the contents of input in that dimension,
    and paddings[D, 1] indicates how many values to add after the contents of
    input in that dimension.

    Both paddings[D, 0] and paddings[D, 1] must be no greater than
    input.dim_size(D) (or input.dim_size(D) - 1)
    if copy_border is true (if false, respectively).

    The padded size of each dimension D of the output is:

    paddings(D, 0) + input.dim_size(D) + paddings(D, 1)
  }];

  let arguments = (ins
    // TODO: add uint8 support when ready.
    TensorOf<[F32, I32, I64]>:$input,
    TensorOf<[I32, I64]>:$pad,
    TFL_MirrorPaddingAttr:$mode
  );

  let results = (outs
    TensorOf<[F32, I32, I64]>:$output
  );

  let hasOptions = 1;
}

def TFL_UniqueOp: TFL_Op<"unique", [NoSideEffect]> {
  let summary = "Unique Op.";

  let description = [{
  This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. In other words:
  }];

  let arguments = (ins
    // TODO: add uint8 support after quantize support.
    TensorOf<[I8, I16, I32, I64, F32]>:$input
  );

  let results = (outs
    TensorOf<[I8, I16, I32, I64, F32]>:$output,
    TensorOf<[I32, I64]>:$idx
  );

  DerivedTFLiteTypeAttr idx_out_type = DerivedTFLiteTypeAttr<[{
    return getResult(1).getType().cast<TensorType>().getElementType().
        cast<IntegerType>().getWidth() > 32 ? tflite::TensorType_INT64 :
            tflite::TensorType_INT32;
    }]>;

  let hasOptions = 1;
}

//===----------------------------------------------------------------------===//
// Quantization ops.
//===----------------------------------------------------------------------===//
def TFL_DequantizeOp: TFL_Op<"dequantize", [NoQuantizableResult]> {
  let summary = "Dequantize operator";

  let description = [{
    Converts quantized array of integers to floating-points according to the
    quantization parameters.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs AnyTensor:$output);
}

def TFL_FakeQuantOp : TFL_Op<"fake_quant", [NoSideEffect]> {
  let summary = "FakeQuant operator";

  let description = [{
    Fake-quantize the 'inputs' tensor of type float via float scalars min and
    max to 'outputs' tensor of same shape as inputs.
  }];

  let arguments = (
    ins AnyTensor:$input,
    // The expected [min, max] range of values.
    F32Attr:$min,
    F32Attr:$max,

    // The bitwidth of the quantization; between 2 and 16, inclusive.
    I32Attr:$num_bits,
    // Quantization range starts from 0 or 1; starts from 1 if true.
    BoolAttr:$narrow_range);

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 0b1;

  let hasOptions = 1;
}

def TFL_QConstOp : Op<TFL_Dialect, "pseudo_qconst", [
    NoSideEffect, FirstAttrDerivedResultType, NoQuantizableResult]> {
  let summary = "Quantized constant pseudo op";

  let description = [{
    Represents a quantized constant value in TensorFlow Lite dialect. This is
    not an actual operation and it will be lowered to buffer instead. The
    quantization parameters are stored as a type attribute in this constant.
  }];

  let arguments = (
    ins TensorTypeAttr:$qtype,
    ElementsAttr:$value
  );

  let results = (outs AnyTensor:$output);

  let builders = [OpBuilder<
    "Builder *, OperationState &state, TypeAttr qtype, Attribute value",
    [{
      state.addAttribute("qtype", qtype);
      state.addAttribute("value", value);
      state.addTypes(qtype.getValue());
    }]>
  ];
}

def TFL_QuantizeOp: TFL_Op<"quantize", [
    FirstAttrDerivedResultType, NoQuantizableResult]> {
  let summary = "Quantize operator";

  let description = [{
    Converts floating point tensors to quantized integer tensors according to
    the quantization parameters defined in the type attribute.
  }];

  let arguments = (
    ins AnyTensor:$input,
    TensorTypeAttr:$qtype
  );

  let results = (outs AnyTensor:$output);
}

def TFL_DensifyOp: TFL_Op<"densify", [NoSideEffect,
                                      SameOperandsAndResultType,
                                      NoQuantizableResult]> {
  let summary = "Densify operator";

  let description = [{
    Converts sparse tensor to dense format.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// LSTM Ops
//===----------------------------------------------------------------------===//

// LSTM Kernel Type attributes
def TFL_LSTM_KT_FULL  : StrEnumAttrCase<"FULL">;
def TFL_LSTM_KT_BASIC  : StrEnumAttrCase<"BASIC">;

def TFL_LSTMKernelTypeAttr : StrEnumAttr<"LSTMKernelType", "lstm kernel type enum",
   [
     TFL_LSTM_KT_FULL,  TFL_LSTM_KT_BASIC
   ]>;

def LstmMandatoryInputsConstraint : PredOpTrait<
  "mandatory operands element types should match",
  // TODO(ashwinm): Replace the indices with input tensor names when that
  // support is available.
  TCopVTEtAreSameAt<[0, 2, 3, 4, 6, 7, 8, 13, 14, 15, 18, 19]>>;

def LstmOptionalPeepholeWeightConstraint : PredOpTrait<
  "the optional peephole weights should all be specified or none",
  TCopVTEtAreSameAt<[9, 10, 11]>>;

def LstmProjectionWeightBiasConstraint : PredOpTrait<
  "either projection weight must be specified or both projection weight and "
  "projection bias must not be specified",
   Or<[
      And<[TypeIsPred<"projection_weights", NoneType>,
           TypeIsPred<"projection_bias", NoneType>]>,
      Neg<TypeIsPred<"projection_weights", NoneType>>]>>;

// TODO(b/137798843): Need to add two additional constraints for both LSTM and
// UnidirectionalSequenceLstm
// For coupling of input and forget gates (cifg): if cifg is true,
// tensor {1, 5, 9, 12, 20} are null; if cifg is
// false, tensors {1, 5, 12} are not null; tensor {9} is not null if
// additionally peephole = true; tensor {20} is not null if additionally layer
// norm = true. For layer norm: if layer norm is false, tensor {20, 21, 22, 23}
// are null; if layer norm is true, tensors {21, 22, 23} are not null; tensor
// {20} is not null if additionally cifg = false.

def LstmResultConstraint : PredOpTrait<
  "the input and result tensor elemental types must be same",
  TCresVTEtIsSameAsOp<0, 0>>;

// This is the basic kernel type LSTM op.
// TODO(b/142417845): Refactor this part to return its tflite node name as
// "lstm".
def TFL_BasicLSTMOp : TFL_Op<"basic_lstm", [NoSideEffect,
    TFL_OperandHasRank<0, 2>, TFL_OperandHasRank<1, 2>, TFL_OperandHasRank<2, 2>,
    TFL_OperandHasRank<3, 1>, TFL_OperandHasRank<4, 2>]> {
  let summary = "The basic lstm operator";

  let description = [{
    basic LSTM Cell Operator.
  }];

  let arguments = (
    ins TensorOf<[F32, I8, QI8, QUI8, QI16, QUI16]>:$data_input,
    TensorOf<[F32, I8, QI8, QUI8, QI16, QUI16]>:$prev_activ_input,
    TensorOf<[F32, I8, QI8, QUI8, QI16, QUI16]>:$weights_input,
    TensorOf<[F32, QI32, QUI32]>:$biases_input,
    TensorOf<[F32, I8, QI8, QUI8, QI16, QUI16]>:$prev_state_input,

    // Attributes
    DefaultValuedAttr<TFL_AFAttr, "TANH">:$fused_activation_function,
    DefaultValuedAttr<F32Attr, "0.0f">:$cell_clip,
    DefaultValuedAttr<F32Attr, "0.0f">:$proj_clip,
    // Since this op is the BASIC kernel only, constrain it.
    Confined<
      DefaultValuedAttr<TFL_LSTMKernelTypeAttr, "BASIC">,
      [TFL_LSTM_KT_BASIC]>:$kernel_type
  );

  let hasOptions = 1;

  let results = (outs 2DTensorOf<[F32, I8, QI8, QUI8, QI16, QUI16]>:$activ_output,
                      2DTensorOf<[F32, I8, QI8, QUI8, QI16, QUI16]>:$state_output,
                      2DTensorOf<[F32, I8, QI8, QUI8, QI16, QUI16]>:$concat_temp,
                      2DTensorOf<[F32, I8, QI8, QUI8, QI16, QUI16]>:$activ_temp);
}

// This is the FULL kernel type LSTM op.
def TFL_LSTMOp :
  TFL_Op<"lstm",
          [LstmMandatoryInputsConstraint,
           LstmOptionalPeepholeWeightConstraint,
           LstmProjectionWeightBiasConstraint,
           LstmResultConstraint,
           TFL_StatefulOp]> {
  let summary = "The full lstm operator";

  let description = [{
Long short-term memory unit (LSTM) recurrent network layer.
The default non-peephole implementation is based on:
http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation,
9(8):1735-1780, 1997.
The peephole implementation is based on:
https://research.google.com/pubs/archive/43905.pdf
Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory
recurrent neural network architectures for large scale acoustic modeling.
INTERSPEECH, 2014.
The coupling of input and forget gate (CIFG) is based on:
http://arxiv.org/pdf/1503.04069.pdf
Greff et al. "LSTM: A Search Space Odyssey"
The layer normalization is based on:
https://arxiv.org/pdf/1607.06450.pdf
Ba et al. “Layer Normalization”
  }];

  let arguments = (
    ins TensorOf<[F32]>:$input,

    // Weights
    TFL_TensorOfOrNone<[F32, I8]>:$input_to_input_weights,
    TensorOf<[F32, I8]>:$input_to_forget_weights,
    TensorOf<[F32, I8]>:$input_to_cell_weights,
    TensorOf<[F32, I8]>:$input_to_output_weights,

    // Recurrent weights
    TFL_TensorOfOrNone<[F32, I8]>:$recurrent_to_input_weights,
    TensorOf<[F32, I8]>:$recurrent_to_forget_weights,
    TensorOf<[F32, I8]>:$recurrent_to_cell_weights,
    TensorOf<[F32, I8]>:$recurrent_to_output_weights,

    // Cell weights
    TFL_TensorOfOrNone<[F32, I8]>:$cell_to_input_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32, I8]>:$cell_to_forget_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32, I8]>:$cell_to_output_weights,

    // Bias
    TFL_TensorOfOrNone<[F32]>:$input_gate_bias,
    TensorOf<[F32]>:$forget_gate_bias,
    TensorOf<[F32]>:$cell_bias,
    TensorOf<[F32]>:$output_gate_bias,

    // Projection weight and bias
    TFL_TensorOfOrNone<[F32, I8]>:$projection_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32]>:$projection_bias,

    // Stateful activation and cell states.
    TFL_StatefulTensor:$input_activation_state,
    TFL_StatefulTensor:$input_cell_state,

    // Layer norm coefficients
    TFL_TensorOfOrNone<[F32]>:$input_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32]>:$forget_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32]>:$cell_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32]>:$output_layer_norm_coefficients,

    // Attributes
    TFL_AFAttr:$fused_activation_function,
    DefaultValuedAttr<F32Attr, "0.0f">:$cell_clip,
    DefaultValuedAttr<F32Attr, "0.0f">:$proj_clip,
    // Since this op is the FULL kernel only, constrain it.
    Confined<
      DefaultValuedAttr<TFL_LSTMKernelTypeAttr, "FULL">,
      [TFL_LSTM_KT_FULL]>:$kernel_type
  );

  let results = (outs AnyTensor:$output);

  // TODO(fengliuai): customize printer and parser to not display
  // empty region.
  let regions = (region AnyRegion:$internal);

  let hasOptions = 1;

  let verifier = [{ return Verify(*this); }];

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    std::vector<int> GetStatefulOperands() { return {18, 19}; }
  }];
}

// UnidirectionalSequenceLstm op.
// TODO(ashwinm): Add constraint to validate the combination of operands
// that are valid for hybrid vs fully quantized vs float only semantics
def TFL_UnidirectionalSequenceLSTMOp :
  TFL_Op<"unidirectional_sequence_lstm",
          [LstmMandatoryInputsConstraint,
           LstmOptionalPeepholeWeightConstraint,
           LstmProjectionWeightBiasConstraint,
           LstmResultConstraint,
           TFL_StatefulOp]> {
  let summary = "Unidirectional sequence lstm operator";

  let description = [{
    A recurrent neural network specified by an LSTM cell. This Op supports
    unrolling the input along the time or batch dimensions, and
    implements the following operation for
    each element in the sequence s = 1...sequence_length:
      outputs[s] = state = activation(LSTMOp(inputs[s]))

    where LSTMOp is LSTM TF Lite Op and the “activation” is the function passed
    as the “fused_activation_function” argument (if not “NONE”).
  }];

  let arguments = (
    ins TensorOf<[F32, I8]>:$input,

    // Weights
    TFL_TensorOfOrNone<[F32, I8]>:$input_to_input_weights,
    TensorOf<[F32, I8]>:$input_to_forget_weights,
    TensorOf<[F32, I8]>:$input_to_cell_weights,
    TensorOf<[F32, I8]>:$input_to_output_weights,

    // Recurrent weights
    TFL_TensorOfOrNone<[F32, I8]>:$recurrent_to_input_weights,
    TensorOf<[F32, I8]>:$recurrent_to_forget_weights,
    TensorOf<[F32, I8]>:$recurrent_to_cell_weights,
    TensorOf<[F32, I8]>:$recurrent_to_output_weights,

    // Cell weights
    TFL_TensorOfOrNone<[F32, I8]>:$cell_to_input_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32, I8]>:$cell_to_forget_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32, I8]>:$cell_to_output_weights,

    // Bias
    TFL_TensorOfOrNone<[F32]>:$input_gate_bias,
    TensorOf<[F32]>:$forget_gate_bias,
    TensorOf<[F32]>:$cell_bias,
    TensorOf<[F32]>:$output_gate_bias,

    // Projection weight and bias
    TFL_TensorOfOrNone<[F32, I8]>:$projection_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32]>:$projection_bias,

    // Stateful activation and cell states.
    TFL_StatefulTensor:$input_activation_state,
    TFL_StatefulTensor:$input_cell_state,

    // Layer norm coefficients
    TFL_TensorOfOrNone<[F32, I8]>:$input_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32, I8]>:$forget_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32, I8]>:$cell_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32, I8]>:$output_layer_norm_coefficients,

    // Attributes
    TFL_AFAttr:$fused_activation_function,
    DefaultValuedAttr<F32Attr, "0.0f">:$cell_clip,
    DefaultValuedAttr<F32Attr, "0.0f">:$proj_clip,
    BoolAttr:$time_major
  );

  let results = (outs AnyTensor:$output);

  let hasOptions = 1;

  let verifier = [{ return Verify(*this); }];

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    std::vector<int> GetStatefulOperands() { return {18, 19}; }
  }];
}

def RnnResultConstraint : PredOpTrait<
  "the input and result tensor elemental types must be same",
  TCresVTEtIsSameAsOp<0, 0>>;

// UnidirectionalSequenceRNN op.
def TFL_UnidirectionalSequenceRNNOp :
  TFL_Op<"unidirectional_sequence_rnn",
         [RnnResultConstraint, TFL_StatefulOp]> {

  let summary = "Unidirectional sequence rnn operator";

  let description = [{
    A recurrent neural network specified by an RNN cell. This Op takes in input
    in a format {batch_size, seq_len, input_size} or
    {seq_len, batch_size, input_size} if it's time-majored.

    It implements the following operation for
    each element in the sequence s = 1...sequence_length:
      outputs[s] = state = activation(RNNOp(inputs[s]))

    where RNNOp is RNNOp TF Lite Op and the “activation” is the function passed
    as the “fused_activation_function” argument (if not “NONE”).
  }];

  let arguments = (
    ins TensorOf<[F32, I8]>:$input,

    // Weights
    TensorOf<[F32, I8]>:$input_to_input_weights,

    // Recurrent weights
    TensorOf<[F32, I8]>:$recurrent_to_input_weights,

    // Bias
    TensorOf<[F32]>:$input_gate_bias,

    // Hidden state.
    TFL_StatefulTensor:$hidden_state,

    // Attributes
    BoolAttr:$time_major,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs TensorOf<[F32, I8]>:$output);

  let hasOptions = 1;

  let customOption = "SequenceRNNOptions";

  let verifier = [{ return Verify(*this); }];

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    std::vector<int> GetStatefulOperands() { return {4}; }
  }];
}

def TFL_WhereOp : TFL_Op<"where", [NoSideEffect]> {
  let summary = "Returns locations of nonzero / true values in a tensor.";

  let description = [{
This operation returns the coordinates of true elements in `condition`. The
coordinates are returned in a 2-D tensor where the first dimension (rows)
represents the number of true elements, and the second dimension (columns)
represents the coordinates of the true elements. Keep in mind, the shape of
the output tensor can vary depending on how many true values there are in
`condition`. Indices are output in row-major order.
  }];

  let arguments = (ins
    I1Tensor:$input
  );

  let results = (outs
    I64Tensor:$index
  );
}

def TFL_NumericVerifyOp : Op<TFL_Dialect, "NumericVerify", [
    SameOperandsShape]> {

  let summary = "Verifies the numericals of the two operands";

  let description = [{
    The NumericVerify op is a debugging op to verify the numericals of the two
    activations. It is a custom op in TFLite.
  }];

  let arguments = (ins
    TensorOf<[QI8, QUI8, QI16, QUI16]>:$input,
    TensorOf<[F32]>:$ref,

    // Attributes
    DefaultValuedAttr<F32Attr, "0.1">:$tolerance
  );

  let results = (outs);
}

def SVDFResultConstraint: PredOpTrait<
  "the input and result tensor elemental types must be same",
  TCresVTEtIsSameAsOp<0, 0>>;

// SVDF op.
def TFL_SVDFOp :
  TFL_Op<"svdf",
         [SVDFResultConstraint, TFL_StatefulOp]> {

  let summary = "Single value decomposition filter operator";

  let description = [{
    The SVDF op is a decomposition of a densely connected op into low rank
    filters.
    For details: https://research.google.com/pubs/pub43813.html
                 https://arxiv.org/abs/1812.02802
  }];

  let arguments = (
    ins TensorOf<[F32, I8]>:$input,

    // Feature Weights.
    TensorOf<[F32, I8]>:$feature_weights,

    // Time weights
    TensorOf<[F32, I8]>:$time_weights,

    // Bias
    TFL_TensorOfOrNone<[F32]>:$input_gate_bias,

    // Activation state.
    TFL_StatefulTensor:$activation_state,

    // Attributes
    I32Attr:$rank,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs TensorOf<[F32, I8]>:$output);

  let hasOptions = 1;

  let verifier = [{ return Verify(*this); }];

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    std::vector<int> GetStatefulOperands() { return {4}; }
  }];
}

def TFL_SegmentSumOp: TFL_Op<"segment_sum", [NoSideEffect]> {
  let summary = "SegmentSum operator";

  let description = [{
    Computes the sum along segments of a tensor.
  }];

  let arguments = (ins
    TensorOf<[F32, I32]>:$data,
    I32Tensor:$segment_ids
  );
  let results = (outs TensorOf<[F32, I32]>:$output);
}

#endif // TFL_OPS
