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

// LINT.IfChange
// This is the operation definition file for TensorFlow Lite.

#ifndef TFL_OPS
#define TFL_OPS

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_interfaces.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td"
include "tensorflow/compiler/mlir/lite/quantization/quantization.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

//===----------------------------------------------------------------------===//
// TFLite dialect string type - uses the TF string type as implementation
//===----------------------------------------------------------------------===//
def TFL_Str : Type<CPred<"$_self.isa<mlir::TF::StringType>()">,
                  "TFLite string type">,
             BuildableType<"getType<mlir::TF::StringType>()">;

//===----------------------------------------------------------------------===//
// TFLite dialect quint8 type - uses the TF quint8 type as implementation
//===----------------------------------------------------------------------===//
def TFL_Quint8 : Type<CPred<"$_self.isa<mlir::TF::Quint8Type>()">,
                    "TFLite quint8 type">,
              BuildableType<"getType<mlir::TF::Quint8Type>()">;

//===----------------------------------------------------------------------===//
// Type that represents control dependencies
//===----------------------------------------------------------------------===//
def TFL_Control: Type<CPred<"$_self.isa<ControlType>()">, "control">,
                 BuildableType<"$_builder.getType<ControlType>()">;


//===----------------------------------------------------------------------===//
// TensorType attribute definitions.
//===----------------------------------------------------------------------===//
// A type attribute containing the TensorType.
def TensorTypeAttr : TypeAttrBase<"TensorType", "Tensor type attribute">;

//===----------------------------------------------------------------------===//
// Derived shape attribute class.
//===----------------------------------------------------------------------===//
class DerivedShapeAttr<code body> : DerivedAttr<"ArrayRef<int64_t>", body>;
class DerivedTFLiteTypeAttr<code body, code convert> :
  DerivedAttr<"tflite::TensorType", body, convert>;

// TFL Runtime op trait predicate.
class TFL_RuntimePredOpTrait<string desc, Pred pred> :
    GenInternalOpTrait<"TFLRuntimeOpTrait"> {
  Pred tflRuntimePredicate = pred;
  string tflRuntimeDescription = desc;
}

class TFL_OperandsHaveSameShapesOrBroadcastableShape<
    list<int> indices, int max_bcast_rank> :
  TFL_RuntimePredOpTrait<"operands do not have the same shape or "
      "broadcastable shapes within the rank " # max_bcast_rank,
    CPred<"TFL::VerifyOperandsHaveSameShapesOrBroadcastableShape("
            "$_op, llvm::ArrayRef<unsigned>({" # !interleave(indices, ", ") #
            "}), " # max_bcast_rank # ")">>;

// These additional types/type constraints here are used to decouple the ops
// from runtime support for the ops. Prefer to use these types when defining
// new TF_Ops for uniformity.

// TFL Runtime type predicate.
class TFL_RuntimeType<TypeConstraint t> {
  Pred tflRuntimeTypePredicate = t.predicate;
  string tflRuntimeTypeDescription = t.summary;
}

class TFL_AnyTypeOf<list<Type> allowedRuntimeTypes, string description = "",
                    list<Type> allowedOpTypes = [AnyType]> :
  AnyTypeOf<allowedOpTypes, description>,
  TFL_RuntimeType<AnyTypeOf<allowedRuntimeTypes, description>>;

class TFL_TensorOf<list<Type> allowedRuntimeTypes,
                   list<Type> allowedOpTypes = [AnyType]> :
  TensorOf<allowedOpTypes>, TFL_RuntimeType<TensorOf<allowedRuntimeTypes>> {
  // Set the summary equal to that representing the runtime types.
  let summary = TensorOf<allowedRuntimeTypes>.summary;
}

class TFL_TensorOfOrNone<list<Type> allowedRuntimeTypes, string description = "",
                         list<Type> allowedOpTypes = [AnyType]> :
  AnyTypeOf<[TFL_TensorOf<allowedOpTypes>, NoneType], description>,
  TFL_RuntimeType<AnyTypeOf<[TFL_TensorOf<allowedRuntimeTypes>, NoneType]>>;

class TFL_VariadicTensorOf<list<Type> allowedRuntimeTypes,
                   list<Type> allowedOpTypes = [AnyType]> :
  Variadic<TensorOf<allowedOpTypes>>,
  TFL_RuntimeType<Variadic<TensorOf<allowedRuntimeTypes>>>;

def TFL_Int32Or64 : SignlessIntOfWidths<[32, 64]>;

def TFL_BoolTensor : TFL_TensorOf<[I1]>;
def TFL_FpTensor : TFL_TensorOf<[F32]>;
def TFL_I32OrI64Tensor : TFL_TensorOf<[TFL_Int32Or64]>;
def TFL_I32Tensor : TFL_TensorOf<[I32]>;
def TFL_I64Tensor : TFL_TensorOf<[I64]>;
def TFL_Complex64Tensor : TFL_TensorOf<[Complex<F<32>>]>;
def TFL_ResourceTensor : TFL_TensorOf<[TF_Resource]>;

// TODO(jpienaar): Expand to all int types.
def TFL_IntTensor : TypeAlias<TFL_I32Tensor, "tensor of any integer type">;

class TFL_0DTensorOf<list<Type> allowedRuntimeTypes,
                     list<Type> allowedOpTypes = [AnyType]> :
  0DTensorOf<allowedOpTypes>, TFL_RuntimeType<TensorOf<allowedRuntimeTypes>>;
class TFL_1DTensorOf<list<Type> allowedRuntimeTypes,
                     list<Type> allowedOpTypes = [AnyType]> :
  1DTensorOf<allowedOpTypes>, TFL_RuntimeType<TensorOf<allowedRuntimeTypes>>;
class TFL_2DTensorOf<list<Type> allowedRuntimeTypes,
                     list<Type> allowedOpTypes = [AnyType]> :
  2DTensorOf<allowedOpTypes>, TFL_RuntimeType<TensorOf<allowedRuntimeTypes>>;

class TFL_1DTensorOfOrNone<list<Type> allowedRuntimeTypes, string description = "",
                         list<Type> allowedOpTypes = [AnyType]> :
  AnyTypeOf<[TensorOf<allowedOpTypes>, NoneType], description>,
  TFL_RuntimeType<AnyTypeOf<[TFL_1DTensorOf<allowedRuntimeTypes>, NoneType]>>;

// This is used to represent the type of "ref tensors" or tensors that are
// used as variables to track state.
def TFL_StatefulTensor : TypeAlias<AnyTensor, "stateful tensor">;

//===----------------------------------------------------------------------===//
// Rank/Shape helpers.
//===----------------------------------------------------------------------===//

// Returns true of operand is none type.
class TFL_OperandIsNoneType<int i> :
  CPred<"$_op.getOperand(" # i # ").getType().isa<NoneType>()">;

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

// Returns true if the n-th operand is ranked and has a dimension length = size
// at the rank dim.
class TFL_OperandDimEquals<int n, int dim, int size> : And<[
  TFL_OperandIsRankedAndHasDimPred<n, dim>,
  CPred<"$_op.getOperand(" # n # ").getType().cast<ShapedType>()"
      ".getShape()[" # dim # " ] == " # size>]>;

// Returns true if the n-th operand is ranked and has a dimension length <=
// size at the rank dim.
class TFL_OperandDimIsAtMost<int n, int dim, int size> : And<[
  TFL_OperandIsRankedAndHasDimPred<n, dim>,
  CPred<"$_op.getOperand(" # n # ").getType().cast<ShapedType>()"
      ".getShape()[" # dim # " ] <= " # size>]>;

// Returns true if the n-th operand has unknown rank or at least rank m.
class TFL_OperandHasAtleastRank<int n, int m> :
  PredOpTrait<"operand " # n # " is " # m # "-D",
    Or<[CPred<"$_op.getOperand(" # n # ").getType().isa<UnrankedTensorType>()">,
      CPred<"$_op.getOperand(" # n #
        ").getType().cast<ShapedType>().getRank() >= " # m>]>>;

class TFL_OperandRankEquals1DimOfOperand<int x, int y> :
  PredOpTrait<"operand " # x # "'s rank equals operand " # y # "'s size",
    Or<[TFL_OperandIsUnrankedPred<x>,
        TFL_OperandIsUnrankedPred<y>,
        CPred<"!$_op.getOperand(" # y #
          ").getType().cast<ShapedType>().hasStaticShape()">,
        CPred<"$_op.getOperand(" # x #
          ").getType().cast<ShapedType>().getRank() == "
          "$_op.getOperand(" # y #
          ").getType().cast<ShapedType>().getShape()[0]">]>>;

class TFL_Operand0DOr1ElementTensor<int x> :
  PredOpTrait<"operand #" # x # " is an 0-d tensor or 1-d tensor w/ 1 element",
    Or<[TFL_OperandHasKnownRank<x, 0>,
        And<[TFL_OperandHasKnownRank<x, 1>, TFL_OperandDimEquals<x, 0, 1>]>]>>;

// Return true if i-th dim of x-th operand is the same as j-th dim of y-th
// operand or any of those operands does not have static shape.
class TFL_OperandsHaveSameDims<int x, int y, int i, int j> :
    Or<[TFL_OperandIsUnrankedPred<x>,
        TFL_OperandIsUnrankedPred<y>,
        CPred<"!$_op.getOperand(" # x #
          ").getType().cast<ShapedType>().hasStaticShape()">,
        CPred<"!$_op.getOperand(" # y #
          ").getType().cast<ShapedType>().hasStaticShape()">,
        CPred<"$_op.getOperand(" # x #
          ").getType().cast<ShapedType>().getShape()[" # i # "] == "
          "$_op.getOperand(" # y #
          ").getType().cast<ShapedType>().getShape()[" # j # "]">]>;

class TFL_OperandsHaveSameDimsTrait<int x, int y, int i, int j> :
  PredOpTrait<"dim " # i # " of operand " # x # " equals to dim " # j #
    " of operand " # y,
    TFL_OperandsHaveSameDims<x, y, i, j>>;

// Return true if number of elements of x-th operand is the same as j-th dim of
// y-th operand or any of those operands does not have static shape.
class TFL_NumElementsEqualsDim<int x, int y, int j> :
  Or<[TFL_OperandIsUnrankedPred<x>,
      TFL_OperandIsUnrankedPred<y>,
      CPred<"!$_op.getOperand(" # x #
        ").getType().cast<ShapedType>().hasStaticShape()">,
      CPred<"!$_op.getOperand(" # y #
        ").getType().cast<ShapedType>().hasStaticShape()">,
      CPred<"$_op.getOperand(" # x #
        ").getType().cast<ShapedType>().getNumElements() == "
        "$_op.getOperand(" # y #
        ").getType().cast<ShapedType>().getShape()[" # j # "]">]>;

class TFL_NumElementsEqualsDimTrait<int x, int y, int j> :
  PredOpTrait<"operand " # x # " has num of elements equals to dim " # j #
    " of operand " # y,
    TFL_NumElementsEqualsDim<x, y, j>>;

// Return true if number of elements of x-th operand equals to n.
class TFL_NumElements<int x, int n> :
  Or<[TFL_OperandIsUnrankedPred<x>,
      CPred<"!$_op.getOperand(" # x #
        ").getType().cast<ShapedType>().hasStaticShape()">,
      CPred<"$_op.getOperand(" # x #
        ").getType().cast<ShapedType>().getNumElements() == " # n>]>;

class TFL_NumElementsTrait<int x, int n> :
  PredOpTrait<"operand " # x # " has num of elements equals to  " # n,
    TFL_NumElements<x, n>>;

// tf.uint8 and tf.quint8 are mapped to the same tflite types, so they are equal
// when used as element types.
class TFL_TFTypesWithSameBits<int i, int j, int num> :
  And<[
    Or<[CPred<"getElementTypeOrSelf($_op.getResult(" # i # ")).isa<mlir::TF::Quint" # num # "Type>()">,
        CPred<"getElementTypeOrSelf($_op.getResult(" # i # ")).isUnsignedInteger(" # num # ")">]>,
    Or<[CPred<"getElementTypeOrSelf($_op.getOperand(" # j # ")).isa<mlir::TF::Quint" # num # "Type>()">,
        CPred<"getElementTypeOrSelf($_op.getOperand(" # j # ")).isUnsignedInteger(" # num # ")">]>]>;

class TFL_TFOperandTypesWithSameBits<int i, int j, int num> :
  And<[
    Or<[CPred<"getElementTypeOrSelf($_op.getOperand(" # i # ")).isa<mlir::TF::Quint" # num # "Type>()">,
        CPred<"getElementTypeOrSelf($_op.getOperand(" # i # ")).isUnsignedInteger(" # num # ")">]>,
    Or<[CPred<"getElementTypeOrSelf($_op.getOperand(" # j # ")).isa<mlir::TF::Quint" # num # "Type>()">,
        CPred<"getElementTypeOrSelf($_op.getOperand(" # j # ")).isUnsignedInteger(" # num # ")">]>]>;

class TFL_OperandIsNoneOrHasRank<int n, int m> :
  PredOpTrait<"operand " # n # " is " # m # "-D",
    Or<[
      TFL_OperandIsNoneType<n>,
      TFL_OperandIsUnrankedPred<n>,
      CPred<"$_op.getOperand(" # n #
      ").getType().cast<ShapedType>().getRank() == " # m>]>>;

class TFL_OperandIsNoneOrHasRankAtMost<int n, int m> :
  PredOpTrait<"operand " # n # " is at most " # m # "-D",
    Or<[
      TFL_OperandIsNoneType<n>,
      TFL_OperandIsUnrankedPred<n>,
      CPred<"$_op.getOperand(" # n #
      ").getType().cast<ShapedType>().getRank() <= " # m>]>>;

class TFL_OperandHasRankAtMostPred<int n, int m> :
  Or<[TFL_OperandIsUnrankedPred<n>,
    CPred<"$_op.getOperand(" # n #
    ").getType().cast<ShapedType>().getRank() <= " # m>]>;

class TFL_OperandHasRankAtMost<int n, int m> :
  PredOpTrait<"operand " # n # " is at most " # m # "-D",
    TFL_OperandHasRankAtMostPred<n, m>>;

class TFL_OperandHasRankAtLeast<int n, int m> :
  PredOpTrait<"operand " # n # " is at least " # m # "-D",
    Or<[TFL_OperandIsUnrankedPred<n>,
      CPred<"$_op.getOperand(" # n #
      ").getType().cast<ShapedType>().getRank() >= " # m>]>>;

class TFL_OperandHasRankRange<int n, int x, int y> :
  PredOpTrait<"operand " # n # " has rank range [" # x # ", " # y # "]",
    Or<[TFL_OperandIsUnrankedPred<n>,
      CPred<"$_op.getOperand(" # n # ").getType().cast<ShapedType>().getRank() "
      ">= " # x # " && $_op.getOperand(" # n # ").getType().cast<ShapedType>()."
      "getRank() <= " # y>]>>;

def TFL_FloatNonNegative : AttrConstraint<
    CPred<"$_self.isa<FloatAttr>() && "
            "!$_self.cast<FloatAttr>().getValue().isNegative()">,
    "whose value is non-negative">;

def TFL_BoolTrue : AttrConstraint<
    CPred<"$_self.isa<BoolAttr>() && $_self.cast<BoolAttr>().getValue()">,
    "whose value is true">;

def TFL_BoolFalse : AttrConstraint<
    CPred<"$_self.isa<BoolAttr>() && !$_self.cast<BoolAttr>().getValue()">,
    "whose value is false">;

class TFL_StringEqualsTo<string value> : AttrConstraint<
    CPred<"$_self.cast<StringAttr>().getValue() == \"" # value # "\"">,
    "whose value equals to '" # value # "'">;

// Ensures the array attribute's size is within the given maximum size.
class TFL_ArrayMaxCount<int n> : AttrConstraint<
    CPred<"$_self.isa<ArrayAttr>() && $_self.cast<ArrayAttr>().size() <= " # n>,
    "whose size is at most " # n>;

// Ensures the given integer attribute has the given value.
class TFL_IntEqualsTo<int n> : AttrConstraint<
    CPred<"$_self.isa<IntegerAttr>() && "
            "$_self.cast<IntegerAttr>().getInt() == " # n>,
    "whose value is " # n>;

// Ensures the given LSTMKernelType attribute has the given value.
class TFL_LSTMKernelTypeEqualsTo<string value> : AttrConstraint<
    CPred<"$_self.isa<LSTMKernelTypeAttr>() && "
            "$_self.cast<LSTMKernelTypeAttr>().getValue() == " # value>,
    "whose value is " # value>;

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

def TFL_SameFirstOperandAndFirstResultElementType :
  PredOpTrait<"values and output must have same element type",
              TFL_TCresVTEtIsSameAsOp<0, 0>>;

// This is a quantization-aware version of TCopVTEtAreSameAt
class TFL_TCopVTEtAreSameAt<int i, int j, int num=8> : Or<[
  TCopVTEtAreSameAt<[i, j]>,
  TFL_TFOperandTypesWithSameBits<i, j, num>,
  And<[
    SubstLeaves<"$_self", "getElementTypeOrSelf($_op.getOperand(" # j # "))",
      quant_QuantizedType.predicate>,
    CPred<"quant::QuantizedType::castToStorageType("
              "getElementTypeOrSelf($_op.getOperand(" # i # "))) == "
          "quant::QuantizedType::castToStorageType("
              "getElementTypeOrSelf($_op.getOperand(" # j # ")))">]>]>;

//===----------------------------------------------------------------------===//
// TFL op common constraints.
//===----------------------------------------------------------------------===//

class OperandsSameElementTypeConstraintBase<string op> :
  PredOpTrait<op # " operands have same element type",
    Or<[
      TCopVTEtIsSameAs<0, 1>,
      // Two operands' values are both quantized and their type have the same
      // underlying storage type.
      And<[
        SubstLeaves<"$_self", "getElementTypeOrSelf($_op.getOperand(0))",
          quant_QuantizedType.predicate>,
        CPred<"quant::QuantizedType::castToStorageType("
                  "getElementTypeOrSelf($_op.getOperand(0))) == "
              "quant::QuantizedType::castToStorageType("
                  "getElementTypeOrSelf($_op.getOperand(1)))">]>]>>;

// This is a constraint for most of the binary ops, e.g., add, mul, div, etc.
// Binary ops lhs & rhs should have the same value type, and is capable to
// compare quantization types as well.
def BinaryOpSameElementTypeConstraint :
  OperandsSameElementTypeConstraintBase<"binary op">;

// This is a constraint for most of the comparison ops, e.g., equal, not_equal,
// greater, greater_equal, less, etc. Comparison ops lhs & rhs should have the
// same value type, and is capable to compare quantization types as well.
def ComparisonOpSameElementTypeConstraint :
  OperandsSameElementTypeConstraintBase<"comparison op">;

//===----------------------------------------------------------------------===//
// TFL common builders.
//===----------------------------------------------------------------------===//

def TFL_BroadcastableBinaryBuilder :
  OpBuilder<(ins "Value":$lhs, "Value":$rhs),
  [{
    auto resultType =
      OpTrait::util::getBroadcastedType(lhs.getType(), rhs.getType());
    if (!resultType)
      mlir::emitError($_state.location, "non-broadcastable operands");
    $_state.addOperands({lhs, rhs});
    $_state.types.push_back(resultType);
  }]>;

def TFL_FusedBroadcastableBinaryBuilder :
  OpBuilder<(ins "Value":$lhs, "Value":$rhs,
    "StringAttr":$fusedActivationFunction),
  [{
    buildFusedBroadcastableBinOp(
       &$_builder, $_state, lhs, rhs, fusedActivationFunction);
  }]>;

def TFL_ComparisonBinaryBuilder :
  OpBuilder<(ins "Value":$lhs, "Value":$rhs),
  [{
    buildComparisonBinOp(&$_builder, $_state, lhs, rhs);
  }]>;

//===----------------------------------------------------------------------===//
// TFL op base class.
//===----------------------------------------------------------------------===//

// LINT.IfChange

class TFL_Op<string mnemonic, list<Trait> traits = []> :
    Op<TFL_Dialect, mnemonic, !listconcat(traits,
      [DeclareOpInterfaceMethods<TFL_RuntimeVerification>])> {
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

class TFL_ConvOp<string mnemonic, string opSummary, int index,
                 list<Trait> additional_traits = []> :
    TFL_Op<mnemonic,[Pure,
                     AccumulatorUniformScale<2, 0, 1>,
                     AffineQuantizedOpInterface,
                     AffineOpCoefficient<index, 1>,
                     QuantizableResult,
                     TFL_SparseOp] # additional_traits> {
  let summary = opSummary # " operator";

  let description = [{
    Performs convolution operation on inputs.

    Inputs:
      `inputs[0]`: required: the input activation tensor
      `inputs[1]`: required: the filter weight tensor
      `inputs[2]`: optional: the bias tensor
  }];

  let results = (outs TFL_TensorOf<[F32, QI8, QUI8, QI16]>:$output);

  let hasOptions = 0b1;
}


//===----------------------------------------------------------------------===//
// TFL op definitions.
//===----------------------------------------------------------------------===//
def TFL_AbsOp : TFL_Op<"abs", [
    Pure,
    QuantizableResult,
    SameOperandsAndResultShape]> {
  let summary = "Absolute value operator";

  let description = [{
Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).
  }];

  let arguments = (ins TFL_TensorOf<[I16, I32, F32, QI8, QI16]>:$x);

  let results = (outs TFL_TensorOf<[I16, I32, F32, QI8, QI16]>:$y);

  let hasFolder = 1;
}

def TFL_AddOp : TFL_Op<"add", [
    TFL_RuntimePredOpTrait<"Operands do not have valid shapes",
      CPred<"TFL::VerifyAddOpShapeConstraints(llvm::cast<AddOp>($_op))">>,
    ResultsBroadcastableShape,
    Pure,
    Commutative,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Addition operator";

  let description = [{
    Element-wise addition operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I16, I32, I64, QI8, QUI8, QI16]>:$lhs,
    TFL_TensorOf<[F32, I16, I32, I64, QI8, QUI8, QI16]>:$rhs,
    TFL_AFAttr:$fused_activation_function);

  let results = (outs
    TFL_TensorOf<[F32, I16, I32, I64, QI8, QUI8, QI16]>:$output);

  let hasFolder = 1;

  let builders = [TFL_FusedBroadcastableBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 1;
}

def TFL_AddNOp : TFL_Op<"add_n", [
    Commutative,
    Pure,
    SameOperandsAndResultsScale,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "add_n operator";

  let description = [{
    Adds all input tensors element-wise.
  }];

  let arguments = (ins
    TFL_VariadicTensorOf<[F32, I32]>:$inputs
  );

  let results = (outs
    TFL_TensorOf<[F32, I32]>:$sum
  );
}

def TFL_ReduceAnyOp : TFL_Op<"reduce_any", [Pure]> {
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
    TFL_BoolTensor:$input,
    TFL_I32Tensor:$reduction_indices,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TFL_BoolTensor:$output
  );

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_ReduceAllOp : TFL_Op<"reduce_all", [Pure]> {
  let summary = [{
Computes the "logical and" of elements across dimensions of a tensor.
  }];

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    TFL_BoolTensor:$input,
    TFL_I32Tensor:$reduction_indices,

    DefaultValuedOptionalAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TFL_BoolTensor:$output
  );

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_TransposeConvOp: TFL_Op<"transpose_conv", [
    Pure,
    TFL_OperandHasRank<0, 1>,
    TFL_OperandHasRank<1, 4>,
    TFL_OperandHasRank<2, 4>,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 2>>,
    AccumulatorUniformScale<3, 1, 2>,
    AffineQuantizedOpInterface, AffineOpCoefficient<0, 1>,
    QuantizableResult,
    TFL_SparseOp,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>,
    DynamicRangeQuantizedOpInterface]> {
  let summary = "Transpose convolution operator";

  let description = [{
    Performs transpose convolution operation on input.
  }];

  let arguments = (ins
    TFL_I32Tensor:$output_shape,
    TFL_TensorOf<[F32, QI8, QUI8, QI16]>:$weights,
    TFL_TensorOf<[F32, QI8, QUI8, QI16]>:$input,
    TFL_TensorOfOrNone<[F32, QI32, I64]>:$bias,
    TFL_PaddingAttr:$padding,
    ConfinedAttr<I32Attr, [IntPositive]>:$stride_h,
    ConfinedAttr<I32Attr, [IntPositive]>:$stride_w,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs TFL_TensorOf<[F32, QI8, QUI8, QI16]>:$output);

  let hasOptions = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 0; }
    int GetQuantizationDimIndex() { return 0; }
    // SparseOpInterface:
    std::vector<int> GetSparseOperands() { return {1}; }
    std::vector<std::vector<int>> GetFloatBlockSize() { return {}; }
    std::vector<std::vector<int>> GetQuantizedBlockSize() { return {}; }
    // DynamicRangeQuantizedOpInterface:
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

def TFL_AveragePool2DOp:
    TFL_Op<"average_pool_2d",
           [Pure,
            SameOperandsAndResultsScale,
            QuantizableResult,
            DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Average_pool_2d operator";

  let description = [{
    Performs average-pooling operation on input.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, QI8, QUI8, QI16]>:$input,
    I32Attr:$filter_height,
    I32Attr:$filter_width,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs TFL_TensorOf<[F32, QI8, QUI8, QI16]>:$output);

  let hasOptions = 1;
  let customOption = "Pool2DOptions";
}

def TFL_ArgMaxOp : TFL_Op<"arg_max", [
    QuantizableResult,
    Pure]> {
  let summary = "ArgMax operator";

  let description = [{
    Returns the index with the largest value across dimensions of a tensor.
  }];

  let arguments = (
    ins TFL_TensorOf<[I1, F32, I32, I8, UI8, QI8, QUI8]>:$input,
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
    }], [{
      TypeAttr::get(getResult().getType().cast<TensorType>().getElementType())
    }]>;
}

def TFL_ArgMinOp : TFL_Op<"arg_min", [
    QuantizableResult,
    Pure]> {
  let summary = "ArgMin operator";

  let description = [{
    Returns the index with the smallest value across dimensions of a tensor.
      a = [1, 10, 26.9, 2.8, 166.32, 62.3]
      b = tf.math.argmin(input = a)
      c = tf.keras.backend.eval(b)
  }];

  let arguments = (
    ins TFL_TensorOf<[I1, F32, I32, I8, UI8, QI8, QUI8]>:$input,
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
    }], [{
      TypeAttr::get(getResult().getType().cast<TensorType>().getElementType())
    }]>;
}

def TFL_CeilOp: TFL_Op<"ceil", [
    Pure,
    TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Ceil operator";

  let description = [{
    Returns element-wise ceil value of the input.
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_ConcatenationOp : TFL_Op<"concatenation",
  [
    Pure,
    TFL_SameFirstOperandAndFirstResultElementType,
    SameOperandsAndResultsScale,
    QuantizableResult
  ]> {
  let summary = "Concatenation operator";

  let description = [{
    Concatenates tensors along one dimension
  }];

  let arguments = (
    ins TFL_VariadicTensorOf<
      [F32, I64, I32, I16, I8, QI8, QUI8, UI8, I1]>:$values,
    I32Attr:$axis,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs
    TFL_TensorOf<
      [F32, I64, I32, I16, I8, QI8, QUI8, UI8, I1]>:$output
  );

  let hasOptions = 1;

  let hasFolder = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // SameScalesOpInterface:
    bool RequiredSameOperandsAndResultsScale(bool sign, int bit_width) {
      // uint8 doesn't require same operands and results scales.
      bool is_uint8 = !sign && (bit_width == 8);
      return !is_uint8;
    }
  }];
}

def TFL_ConstOp : Op<TFL_Dialect, "pseudo_const", [ConstantLike, Pure,
    FirstAttrDerivedResultType,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {
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
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "TypedAttr":$value),
    [{
      $_state.addAttribute("value", value);
      $_state.addTypes(value.getType());
    }]>
  ];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

def TFL_SparseConstOp : Op<TFL_Dialect, "pseudo_sparse_const", [
    Pure,
    FirstAttrDerivedResultType,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {
  let summary = "Sparse constant pseudo op.";

  let description = [{
    Represents a sparse constant value in TensorFlow Lite dialect. This is not
    an actual operation and it will be lowered to buffer instead.
  }];

  let arguments = (ins ElementsAttr:$value,
                   SparsityParameterAttr:$s_param,
                   ElementsAttr:$compressed_data);

  let results = (outs AnyTensor:$output);

  let builders = [
    OpBuilder<(ins "TypedAttr":$value, "SparsityParameterAttr":$s_param,
      "Attribute":$compressed_data),
    [{
      $_state.addTypes(value.getType());
      $_state.addAttribute("value", value);
      $_state.addAttribute("s_param", s_param);
      $_state.addAttribute("compressed_data", compressed_data);
    }]>
  ];
}

def TFL_ExternalConstOp : Op<TFL_Dialect, "external_const", [
    Pure,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {
  let summary = "External const op.";

  let description = [{
    External const op holds a `buffer_index` which points to a constant
    in the flatbuffer.
  }];

  let arguments = (ins I32Attr:$buffer_index);

  let results = (outs AnyTensor:$output);
}

def TFL_Conv2DOp : TFL_ConvOp<"conv_2d", "Convolution", 0,
      [DeclareOpInterfaceMethods<InferTypeOpInterface>,
       DeclareOpInterfaceMethods<TFL_ArithmeticCount>,
       DynamicRangeQuantizedOpInterface]> {
  let arguments = (
    ins TFL_TensorOf<[F32, QI8, QUI8, QI16]>:$input,
    TFL_TensorOf<[F32, QI4, QI8, QUI8]>:$filter,
    TFL_1DTensorOfOrNone<[F32, I32, I64]>:$bias,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    TFL_AFAttr:$fused_activation_function,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w
  );

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 0; }
    int GetQuantizationDimIndex() { return 0; }
    // SparseOpInterface:
    std::vector<int> GetSparseOperands() { return {1}; }
    std::vector<std::vector<int>> GetFloatBlockSize() { return {}; }
    std::vector<std::vector<int>> GetQuantizedBlockSize() { return {}; }

    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);

    // DynamicRangeQuantizedOpInterface:
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

def TFL_CosOp: TFL_Op<"cos", [
    Pure,
    TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Cosine operator";

  let description = [{
    Computes element-wise Cosine of input
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }

  }];
}

def TFL_CumsumOp: TFL_Op<"cumsum", [
    Pure,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandHasRank<1, 0>]> {
  let summary = "Cumsum operator";

  let description = [{
    Compute the cumulative sum of the tensor x along axis.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I32, I64]>:$input,
    TFL_I32Tensor:$axis,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$exclusive,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$reverse
  );

  let results = (outs TFL_TensorOf<[F32, I32, I64]>:$output);

  let hasOptions = 1;
}

def TFL_DepthwiseConv2DOp :
    TFL_ConvOp<"depthwise_conv_2d", "Depthwise-separable convolution", 3,
                [DeclareOpInterfaceMethods<TFL_ArithmeticCount>,
                 DynamicRangeQuantizedOpInterface]> {
  let arguments = (
    ins TFL_TensorOf<[F32, QI8, QUI8, QI16]>:$input,
    TFL_TensorOf<[F32, QI4, QI8, QUI8]>:$filter,
    TFL_1DTensorOfOrNone<[F32, I32, I64]>:$bias,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    TFL_AFAttr:$fused_activation_function,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    I32Attr:$depth_multiplier
  );

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 3; }
    int GetQuantizationDimIndex() { return 3; }
    // SparseOpInterface:
    std::vector<int> GetSparseOperands() { return {1}; }
    std::vector<std::vector<int>> GetFloatBlockSize() { return {}; }
    std::vector<std::vector<int>> GetQuantizedBlockSize() { return {}; }
    // DynamicRangeQuantizedOpInterface:
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

// TODO(jpienaar): Update post discussion on semantics of FC OP.
def TFL_FullyConnectedOp : TFL_Op<"fully_connected", [
    Pure, AccumulatorUniformScale<2, 0, 1>,
    AffineQuantizedOpInterface,
    AffineOpCoefficient<-1, 1>,
    TFL_SparseOp,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>,
    QuantizableResult,
    DynamicRangeQuantizedOpInterface]> {
  let summary = "Fully connected op";

  let arguments = (ins
    TFL_TensorOf<[F32, QI8, QUI8, QI16, QUI16]>:$input,
    TFL_TensorOf<[F32, QI4, QI8, QUI8, QI16]>:$filter,
    TFL_TensorOfOrNone<[F32, QI32, QUI32]>:$bias,

    TFL_AFAttr:$fused_activation_function,
    TFL_FullyConnectedOptionsWeightFormatAttr:$weights_format,
    BoolAttr:$keep_num_dims,
    // Used in post-training dynamic range quantization. If the value is true,
    // input activations are asymmetrically quantized.
    OptionalAttr<BoolAttr>:$asymmetric_quantize_inputs
  );

  // Depending on the weights format, this op can have one or two outputs.
  let results = (outs
    TFL_VariadicTensorOf<[F32, QI8, QUI8, QI16, QUI16]>:$output
  );

  let hasVerifier = 1;

  let hasOptions = 1;

  let hasCanonicalizer = 1;

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 0; }
    int GetQuantizationDimIndex() { return -1; }
    // SparseOpInterface:
    std::vector<int> GetSparseOperands() { return {1}; }
    std::vector<std::vector<int>> GetFloatBlockSize() { return {{1, 4}}; }
    std::vector<std::vector<int>> GetQuantizedBlockSize() { return {{1, 16}}; }
    // DynamicRangeQuantizedOpInterface:
    bool RequireAsymmetricQuantizeInputsAttr() { return true; }
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

def TFL_BatchMatMulOp : TFL_Op<"batch_matmul", [
   Pure,
   TFL_OperandHasAtleastRank<0, 2>,
   TFL_OperandHasAtleastRank<1, 2>,
   QuantizableResult,
   PredOpTrait<"x and output must have same element type or they are int8 and int32",
       Or<[TFL_TCresVTEtIsSameAsOp<0, 0>,
           And<[CPred<"getElementTypeOrSelf($_op.getOperand(0)).isInteger(8)">,
                CPred<"getElementTypeOrSelf($_op.getOperand(1)).isInteger(8)">,
                CPred<"getElementTypeOrSelf($_op.getResult(0)).isInteger(32)">]>]>>,
   TFL_RuntimePredOpTrait<"lhs and rhs of this op must have rank between [2, 5]",
     And<[TFL_OperandHasRankAtMostPred<0, 5>,
          TFL_OperandHasRankAtMostPred<1, 5>]>>,
   DynamicRangeQuantizedOpInterface]> {

  let summary = "Batch Matrix Multiply Operator";

  let description = [{
Performs a batched matrix multiplication on the inputs. Follows the
conventions of TensorFlow BatchMatMulV2, with support for unknown dimensions
in the batch dimensions and broadcasting.

    Inputs:
      `inputs[0]`: required: input LHS
      `inputs[1]`: required: input RHS
      `adjoint_lhs`: optional: Transpose LHS (default false)
      `adjoint_lhs`: optional: Transpose LHS (default false)
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, QI8, QI16, I8]>:$x,
    TFL_TensorOf<[F32, QI8, QI16, I8]>:$y,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$adj_x,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$adj_y,
    // Used in post-training dynamic range quantization. If the value is true,
    // input activations are asymmetrically quantized.
    OptionalAttr<BoolAttr>:$asymmetric_quantize_inputs
  );

   let results = (outs
    TFL_TensorOf<[F32, QI8, QI16, I32]>:$output
  );

  let hasOptions = 1;

  let extraClassDeclaration = [{
    // DynamicRangeQuantizedOpInterface:
    bool RequireAsymmetricQuantizeInputsAttr() { return true; }
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

def TFL_GatherOp : TFL_Op<"gather", [
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale,
    TFL_OperandHasAtleastRank<0, 1>,
    PredOpTrait<"params and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    DynamicRangeQuantizedOpInterface
  ]> {
  let summary = "Gather operator";

  let description = [{
    Gather slices from `params` axis `axis` according to `indices`.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I1, I8, I16, I32, I64, TFL_Str, UI8, QI8, QUI8, QI16]>:$params,
    TFL_TensorOf<[I16, I32, I64]>:$indices,
    I32Attr:$axis,
    DefaultValuedOptionalAttr<I32Attr, "0">:$batch_dims
  );

  let builders =
  [
    OpBuilder<(ins "Value":$params, "Value":$indices, "IntegerAttr":$axis, "IntegerAttr":$batch_dims),
    [{ BuildGatherOp(&$_builder, $_state, params, indices, axis, batch_dims); }]>
  ];

  let results = (outs
    TFL_TensorOf<[F32, I1, I8, I16, I32, I64, TFL_Str, UI8, QI8, QUI8, QI16]>:$output
  );

  let hasOptions = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // DynamicRangeQuantizedOpInterface:
    std::vector<int> GetQuantizableOperandIndices() { return {0}; }
  }];
}

def TFL_GatherNdOp : TFL_Op<"gather_nd", [
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale,
    PredOpTrait<"params and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "Gather_nd operator";

  let description = [{
    Gather slices from `params` into a Tensor with shape specified by `indices`.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I8, I16, I64, I32, UI8, TFL_Str]>:$params,
    TFL_TensorOf<[I16, I32, I64]>:$indices
  );

  let results = (outs
    TFL_TensorOf<[F32, I8, I16, I64, I32, UI8, TFL_Str]>:$output
  );
}

def TFL_ScatterNdOp : TFL_Op<"scatter_nd", [
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale,
    TFL_OperandHasAtleastRank<0, 1>,
    TFL_OperandHasAtleastRank<1, 1>,
    PredOpTrait<"updates and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 1>>
  ]> {
  let summary = "Scatter_nd operator";

  let description = [{
    Scatter `updates` into a new tensor according to `indices`
  }];

  let arguments = (ins
    TFL_TensorOf<[I32]>:$indices,
    TFL_TensorOf<[F32, I8, I64, I32, UI8, I1]>:$updates,
    TFL_1DTensorOf<[I32]>:$shape
  );

  let results = (outs
    TFL_TensorOf<[F32, I8, I64, I32, UI8, I1]>:$output
  );

  let hasVerifier = 1;

  let hasOptions = 1;
}

// Same type check of lhs and rhs is handled by the ResultsBroadcastableShape trait.
def TFL_LessEqualOp : TFL_Op<"less_equal", [
    ResultsBroadcastableShape,
    QuantizableResult,
    ComparisonOpSameElementTypeConstraint,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    Pure]> {
  let summary = "Less_equal operator";

  let description = [{
    Element-wise less_equal operation.
  }];

  let arguments = (
      ins TFL_TensorOf<[F32, I32, I64, QI8, QUI8]>:$lhs,
      TFL_TensorOf<[F32, I32, I64, QI8, QUI8]>:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 0;
}

def TFL_LocalResponseNormalizationOp : TFL_Op<"local_response_normalization", [
    TFL_OperandHasRank<0, 4>,
    TF_SameOperandsAndResultTypeResolveRef,
    Pure]> {
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
      TFL_FpTensor:$input,
      I32Attr:$radius,
      F32Attr:$bias,
      F32Attr:$alpha,
      F32Attr:$beta
  );

  let results = (outs
    TFL_FpTensor:$output
  );

  let hasOptions = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_GreaterEqualOp : TFL_Op<"greater_equal", [
    QuantizableResult,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    ResultsBroadcastableShape,
    ComparisonOpSameElementTypeConstraint,
    Pure]> {
  let summary = "Greater_equal operator";

  let description = [{
    Element-wise greater_equal operation.
  }];

  let arguments = (
      ins TFL_TensorOf<[F32, I16, I32, I64, QUI8, QI8]>:$lhs,
      TFL_TensorOf<[F32, I16, I32, I64, QUI8, QI8]>:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 0;
}

def TFL_MatrixDiagOp : TFL_Op<"matrix_diag", [
  QuantizableResult,
  Pure,
  TFL_OperandHasAtleastRank<0, 1>,
  PredOpTrait<"operand and result must have the same element type",
    TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
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
    TFL_TensorOf<[F32, I8, I16, I32, I64, UI8, QUI8, QI8, TFL_Quint8]>:$diagonal
  );

  let results = (outs
    TFL_TensorOf<[F32, I8, I16, I32, I64, UI8, QUI8, QI8, TFL_Quint8]>:$output
  );

  let hasOptions = 0;
}

def TFL_MatrixSetDiagOp : TFL_Op<"matrix_set_diag", [
    QuantizableResult,
    TFL_OperandHasAtleastRank<0, 2>,
    PredOpTrait<"input and result must have the same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure]> {
  let summary = [{
    Returns a batched matrix tensor with new batched diagonal values.
  }];

  let description = [{
Given `input` and `diagonal`, this operation returns a tensor with the
same shape and values as `input`, except for the main diagonal of the
innermost matrices.  These will be overwritten by the values in `diagonal`.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I8, I16, I32, I64, UI8, QI8, QI16, QUI8, TFL_Quint8]>:$input,
    TFL_TensorOf<[F32, I8, I16, I32, I64, UI8, QI8, QI16, QUI8, TFL_Quint8]>:$diagonal
  );

  let results = (outs
    TFL_TensorOf<[F32, I8, I16, I32, I64, UI8, QI8, QI16, QUI8, TFL_Quint8]>:$result
  );

  let hasOptions = 0;
}

// These ops are named NonMaxSuppressionV4 & NonMaxSuppressionV5 to be
// consistent with TensorFlow's naming. They are NOT 'versions' of NMS in the
// sense that one is an incremental change over the other.
// In reality NonMaxSuppressionV5 implements Soft Non Max Suppression and
// NonMaxSuppressionV4 performs hard NMS.

def TFL_NonMaxSuppressionV4Op : TFL_Op<"non_max_suppression_v4", [
  Pure,
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
    TFL_I32Tensor:$max_output_size,
    TFL_FpTensor:$iou_threshold,
    TFL_FpTensor:$score_threshold
  );

  let results = (outs
    TFL_I32Tensor:$selected_indices,
    TFL_I32Tensor:$valid_outputs
  );
}

def TFL_NonMaxSuppressionV5Op : TFL_Op<"non_max_suppression_v5", [
  Pure,
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
    TFL_I32Tensor:$max_output_size,
    TFL_FpTensor:$iou_threshold,
    TFL_FpTensor:$score_threshold,
    TFL_FpTensor:$soft_nms_sigma
  );

  let results = (outs
    TFL_I32Tensor:$selected_indices,
    TFL_FpTensor:$selected_scores,
    TFL_I32Tensor:$valid_outputs
  );
}

def TFL_NotEqualOp : TFL_Op<"not_equal", [
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    ComparisonOpSameElementTypeConstraint,
    ResultsBroadcastableShape,
    Commutative,
    Pure]> {
  let summary = "Not_equal operator";

  let description = [{
    Element-wise not_equal operation.
  }];

  let arguments = (
      ins TFL_TensorOf<[I1, F32, I32, I64, QUI8, QI8, TFL_Quint8, TFL_Str]>:$lhs,
      TFL_TensorOf<[I1, F32, I32, I64, QUI8, QI8, TFL_Quint8, TFL_Str]>:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let builders =
  [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs),
    [{
        buildComparisonBinOp(&$_builder, $_state, lhs, rhs);
      }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];
}

def TFL_DivOp : TFL_Op<"div", [
    // TODO(fengliuai): NoQuantizableResult is only correct for int8
    // quantization. update to handle Uint8 quantization.
    BinaryOpSameElementTypeConstraint,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 5>,
    ResultsBroadcastableShape,
    Pure,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Division operator";

  let description = [{
    Element-wise division operation.
  }];

  let arguments = (
      ins TFL_TensorOf<[F32, I32, QUI8]>:$lhs,
      TFL_TensorOf<[F32, I32, QUI8]>:$rhs,
      TFL_AFAttr:$fused_activation_function);

  let results = (outs TFL_TensorOf<[F32, I32, QUI8]>:$output);

  let builders = [TFL_FusedBroadcastableBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 1;

  let hasFolder = 1;
}

def TFL_EluOp: TFL_Op<"elu", [
    Pure,
    SameOperandsAndResultShape,
    TFL_SameFirstOperandAndFirstResultElementType]> {
  let summary = "Exponential Linear Unit operator";
  let description = [{
    Computes the exponential linear
      f(x) -> exp(x) - 1 for x < 0, x for x >= 0.
    element-wise.
  }];

  let arguments = (ins TFL_TensorOf<[F32, I8]>:$x);

  let results = (outs TFL_TensorOf<[F32, I8]>:$y);

  let hasOptions = 0;
}

def TFL_EmbeddingLookupOp: TFL_Op<"embedding_lookup",
    [Pure,
     PredOpTrait<"value and output must have same element type",
       TFL_TCresVTEtIsSameAsOp<0, 1>>,
     TFL_OperandHasRank<0, 1>,
     TFL_OperandHasRankAtLeast<1, 2>,
     DynamicRangeQuantizedOpInterface,
     QuantizableResult
    ]> {
  let summary = "Embedding lookup operator";

  let description = [{
    Looks up ids in a list of embedding tensors.
  }];

  let arguments = (ins
    TFL_TensorOf<[I32]>:$lookup,
    TFL_TensorOf<[F32, I8, UI8]>:$value
   );

  let results = (outs TFL_TensorOf<[F32, I8, UI8]>:$output);

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // DynamicRangeQuantizedOpInterface:
    std::vector<int> GetQuantizableOperandIndices() { return {1}; }
  }];
}

def TFL_EqualOp: TFL_Op<"equal", [
    Commutative,
    Pure,
    ResultsBroadcastableShape,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    QuantizableResult,
    ComparisonOpSameElementTypeConstraint]> {
  let summary = "Equal operator";

  let description = [{
    Returns the truth element of x == y element-wise
  }];

  let arguments = (
    ins
    TFL_TensorOf<[I1, F32, I16, I32, I64, QI8, QUI8, UI8, TFL_Str]>:$x,
    TFL_TensorOf<[I1, F32, I16, I32, I64, QI8, QUI8, UI8, TFL_Str]>:$y
  );

  let results = (outs TFL_BoolTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];
}

def TFL_ExpOp: TFL_Op<"exp", [
    Pure,
    QuantizableResult,
    TFL_SameFirstOperandAndFirstResultElementType]> {
  let summary = "Natural exponentiation operator";

  let description = [{
    Performs element-wise natural exponentiation operation on input.
  }];

  let arguments = (ins TFL_TensorOf<[F32, QI8, QI16]>:$x);

  let results = (outs TFL_TensorOf<[F32, QI8, QI16]>:$y);

  // This builder doesn't work with quantized type, so it can only be used by
  // non-quantization tablegen patterns. Currently, it is used by the
  // elementwise-move reordering pattern in the optimize_patterns.td
  let builders = [
    OpBuilder<(ins "Value":$input),
    [{
      $_state.addOperands({input});
      $_state.addTypes(input.getType());
    }]>
  ];

  let hasOptions = 0b1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_ExpandDimsOp: TFL_Op<"expand_dims", [
    Pure,
    SameOperandsAndResultsScale,
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
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
  let arguments = (ins AnyTensor:$input, TFL_I32OrI64Tensor:$dim);

  let results = (outs AnyTensor:$output);

  let hasOptions = 1;
}

def TFL_SqueezeOp: TFL_Op<"squeeze", [Pure,
                                      QuantizableResult,
                                      SameOperandsAndResultsScale]> {
  let summary = "Removes dimensions of size 1 from the shape of a tensor.";

  let description = [{
Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`squeeze_dims`.

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
    ConfinedAttr<DefaultValuedOptionalAttr<I64ArrayAttr, "{}">, [TFL_ArrayMaxCount<8>]>:$squeeze_dims
  );

  let results = (outs
    AnyTensor:$output
  );

  let hasFolder = 1;
  let hasOptions = 1;

  let customOption = "SqueezeOptions";
}

def TFL_FillOp: TFL_Op<"fill", [
    Pure,
    SameOperandsAndResultsScale,
    QuantizableResult,
    PredOpTrait<"input and result must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 1>>]> {
  let summary = "Fill the tensor with given value.";
  let description = [{
    Fill the tensor with given value.
  }];

  let arguments = (ins TFL_I32OrI64Tensor:$dims,
                   TFL_TensorOf<[F32, F16, I32, I64, I1, QI8, QI16, TFL_Str]>:$input);

  let results = (outs TFL_TensorOf<[F32, F16, I32, I64, I1, QI8, QI16, TFL_Str]>:$result);

  let hasOptions = 0;
}

def TFL_FloorOp: TFL_Op<"floor", [
    Pure,
    TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Floor operator";

  let description = [{
    Returns element-wise floor value of the input.
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_FloorDivOp : TFL_Op<"floor_div", [
    ResultsBroadcastableShape,
    Pure,
    BinaryOpSameElementTypeConstraint,
    PredOpTrait<"lhs and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>]> {
  let summary = "Floor div operator";

  let description = [{
    Element-wise floor div operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I8, I16, I32]>:$lhs,
    TFL_TensorOf<[F32, I8, I16, I32]>:$rhs);

  let results = (outs TFL_TensorOf<[F32, I8, I16, I32]>:$output);

  let builders = [TFL_BroadcastableBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];
}

def TFL_FloorModOp : TFL_Op<"floor_mod", [
    ResultsBroadcastableShape,
    Pure,
    BinaryOpSameElementTypeConstraint,
    PredOpTrait<"lhs and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>]> {
  let summary = "Division reminder";

  let description = [{
    Element-wise division reminder operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[I8, I16, I32, I64, F32]>:$lhs,
    TFL_TensorOf<[I8, I16, I32, I64, F32]>:$rhs);

  let results = (outs TFL_TensorOf<[I8, I16, I32, I64, F32]>:$output);

  let builders = [TFL_BroadcastableBinaryBuilder];
}

def TFL_GreaterOp : TFL_Op<"greater", [
    ResultsBroadcastableShape,
    ComparisonOpSameElementTypeConstraint,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    QuantizableResult,
    Pure]> {
  let summary = "Greater operator";

  let description = [{
    Element-wise greater operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I32, I64, QUI8, QI8, TFL_Quint8]>:$lhs,
    TFL_TensorOf<[F32, I32, I64, QUI8, QI8, TFL_Quint8]>:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];
}

def TFL_HardSwishOp: TFL_Op<"hard_swish", [
    Pure,
    SameOperandsAndResultShape,
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "Hardswish activation function.";
  let description = [{
    Computes hard-swish activation function
      f(x) -> (x * relu6(x+3))/6
    element-wise.
  }];

  let arguments = (ins TFL_TensorOf<[F32, QUI8, QI8]>:$input);

  let results = (outs TFL_TensorOf<[F32, QUI8, QI8]>:$output);

  let hasOptions = 0;
}

def TFL_L2NormalizationOp : TFL_Op<"l2_normalization", [Pure,
    FixedOutputRangeInterface,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "L2 Normalize Operator";

  let description = [{
    L2Normalization Op
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, QUI8, QI8, QUI16, QI16, I8]>:$input,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs TFL_TensorOf<[F32, QUI8, QI8, QUI16, QI16, I8]>:$output);

  let hasOptions = 1;

  let customOption = "L2NormOptions";

  let extraClassDeclaration = [{
  // FixedOutputRangeInterface:
  quant::UniformQuantizedType GetFixedOutputRange(
      bool is_signed, int bit_width) {
    auto result_type = getOutput().getType();
    // central_value = min_value / 2 + (max_value - 1) / 2 + 1
    // zero_point = central_value
    // scale = 1. / (central_value - min_value)
    return quant::GetFixedOutputRange(is_signed, bit_width, result_type,
        /*scale=*/1.0 / (1<<(bit_width-1)), /*zero_point=*/0);
  }
  }];
}

def TFL_LeakyReluOp: TFL_Op<"leaky_relu", [
    SameOperandsAndResultShape,
    QuantizableResult,
    Pure,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "Leaky Relu operator";

  let description = [{
    Element-wise Leaky ReLU operator
      x -> x >= 0 ? x : (alpha * x)
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, QUI8, QI8, TFL_Quint8, QI16]>:$input,
    // Slope of the activation function at x < 0.
    F32Attr:$alpha
  );

  let results = (outs TFL_TensorOf<[F32, QUI8, QI8, TFL_Quint8, QI16]>:$output);

  let hasOptions = 0b1;
}

def TFL_LessOp : TFL_Op<"less", [
    ResultsBroadcastableShape,
    ComparisonOpSameElementTypeConstraint,
    QuantizableResult,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    Pure]> {
  let summary = "Less operator";

  let description = [{
    Element-wise less operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I16, I32, I64, QUI8, QI8, TFL_Quint8]>:$lhs,
    TFL_TensorOf<[F32, I16, I32, I64, QUI8, QI8, TFL_Quint8]>:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let builders = [TFL_ComparisonBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];
}

def TFL_LogicalAndOp : TFL_Op<"logical_and", [Pure]> {
  let summary = "Logical AND operator";

  let description = [{
    Element-wise logical AND operation.
  }];

  let arguments = (
    ins TFL_BoolTensor:$lhs,
    TFL_BoolTensor:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];
}

def TFL_LogicalNotOp : TFL_Op<"logical_not", [
    Pure,
    SameOperandsAndResultShape]> {
  let summary = "Logical NOT operator";

  let description = [{
    Element-wise logical NOT operation.
  }];

  let arguments = (ins TFL_BoolTensor:$lhs);

  let results = (outs TFL_BoolTensor:$output);
}

def TFL_LogicalOrOp : TFL_Op<"logical_or", [Pure]> {
  let summary = "Logical OR operator";

  let description = [{
    Element-wise logical OR operation.
  }];

  let arguments = (
    ins TFL_BoolTensor:$lhs,
    TFL_BoolTensor:$rhs);

  let results = (outs TFL_BoolTensor:$output);

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];
}

def TFL_LogisticOp: TFL_Op<"logistic", [
    Pure,
    PredOpTrait<"x and y must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    SameOperandsAndResultShape,
    FixedOutputRangeInterface,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Logistic operator";

  let description = [{
    Computes element-wise Sigmoid of input
  }];

  let arguments = (ins TFL_TensorOf<[F32, QI8, QUI8, QI16, TFL_Quint8]>:$x);

  let results = (outs TFL_TensorOf<[F32, QI8, QUI8, QI16, TFL_Quint8]>:$y);

  let extraClassDeclaration = [{
  // FixedOutputRangeInterface:
  quant::UniformQuantizedType GetFixedOutputRange(
      bool is_signed, int bit_width) {
    auto result_type = getY().getType();
    // zero_point = 0
    // scale = 1. / (max_value + 1)
    return quant::GetFixedOutputRange(is_signed, bit_width, result_type,
        /*scale=*/1.0 / (1<<(bit_width)),
        /*zero_point=*/-(1<<(bit_width-1)));
  }
  }];

  // This builder doesn't work with quantized type, so it can only be used by
  // non-quantization tablegen patterns. Currently, it is used by the
  // elementwise-move reordering pattern in the optimize_patterns.td
  let builders = [
    OpBuilder<(ins "Value":$input),
    [{
      $_state.addOperands({input});
      $_state.addTypes(input.getType());
    }]>
  ];
}

def TFL_LogOp: TFL_Op<"log", [
    Pure,
    TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Natural logarithm operator";

  let description = [{
    Performs element-wise natural logarithm operation on input.
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_LogSoftmaxOp : TFL_Op<"log_softmax", [
    Pure,
    SameOperandsAndResultShape,
    PredOpTrait<"x and y must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    FixedOutputRangeInterface,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Log softmax operator";

  let description = [{
    Computes element-wise log softmax activations with the following formula

      input - log(reduce_sum(exp(input), dim))
  }];

  let arguments = (ins TFL_TensorOf<[F32, QUI8, QI8, TFL_Quint8]>:$input);

  let results = (outs TFL_TensorOf<[F32, QUI8, QI8, TFL_Quint8]>:$output);

  let hasOptions = 1;

  let extraClassDeclaration = [{
  // FixedOutputRangeInterface:
  quant::UniformQuantizedType GetFixedOutputRange(
      bool is_signed, int bit_width) {
    auto result_type = getOutput().getType();
    // zero_point = max_value
    // scale = -log_softmax_output_min / (max_value + 1)
    return quant::GetFixedOutputRange(is_signed, bit_width, result_type,
        /*scale=*/16.0 / 256, /*zero_point=*/127);
  }
  }];
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
    TFL_OperandHasRank<0, 4>,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    MaxPoolOperandAndResultConstraints,
    SameOperandsAndResultsScale,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Max Pool 2D op";

  let description = [{
    Performs max pool 2D on input.

    Inputs:
      `inputs[0]`: required: the input tensor
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, QUI8, QI8, QI16, TFL_Quint8]>:$input,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_w,
    I32Attr:$stride_h,
    I32Attr:$filter_width,
    I32Attr:$filter_height,
    TFL_AFAttr:$fused_activation_function
  );

  let results = (outs TFL_TensorOf<[F32, QUI8, QI8, QI16, TFL_Quint8]>:$output);

  let hasOptions = 1;

  let customOption = "Pool2DOptions";
}

def TFL_MaximumOp : TFL_Op<"maximum", [
    ResultsBroadcastableShape,
    Pure,
    QuantizableResult,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 5>,
    Commutative,
    SameOperandsAndResultsScale]> {
  let summary = "Max operator";
  let description = [{
    Element-wise max operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, TFL_Int32Or64, QI8, QUI8, QI16]>:$lhs,
    TFL_TensorOf<[F32, TFL_Int32Or64, QI8, QUI8, QI16]>:$rhs
  );

  let results = (outs
    TFL_TensorOf<[F32, TFL_Int32Or64, QI8, QUI8, QI16]>:$max
  );

  let builders = [TFL_BroadcastableBinaryBuilder];

  let hasOptions = 0;
}

def TFL_MeanOp : TFL_Op<"mean", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    QuantizableResult,
    Pure]> {
  let summary = "Mean operator";

  let description = [{
    Computes the mean of elements across dimensions of a tensor.
    Reduces input_tensor along the dimensions given in axis.
    Unless keepdims is true, the rank of the tensor is reduced by 1 for
    each entry in axis. If keepdims is true, the reduced dimensions are retained
    with length 1.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, UI8, QI16]>:$input,
    TFL_TensorOf<[I32, I64]>:$axis,
    BoolAttr:$keep_dims
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, UI8, QI16]>:$output);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_OneHotOp : TFL_Op<"one_hot", [
    QuantizableResult,
    Pure]> {
  let summary = "OneHot operator";

  let description = [{
    Returns a one-hot tensor.The locations represented by indices in `indices`
    take value `on_value`, while all other locations take value `off_value`.

    If the input `indices` is rank `N`, the output will have rank `N+1`,
    The new axis is created at dimension `axis` (default: the new axis is
    appended at the end).
  }];

  let arguments = (ins
    TFL_TensorOf<[I32, I64]>:$indices,
    TFL_I32Tensor:$depth,
    TFL_TensorOf<[F32, I32, I64, I1, I8, UI8]>:$on_value,
    TFL_TensorOf<[F32, I32, I64, I1, I8, UI8]>:$off_value,

    I32Attr:$axis
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, I1, I8, UI8]>:$output
  );

  let hasOptions = 1;
}

def TFL_RoundOp: TFL_Op<"round", [
    Pure,
    TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Round operator";

  let description = [{
Rounds the values of a tensor to the nearest integer, element-wise.
  }];

  let arguments = (ins
    TFL_FpTensor:$x
  );

  let results = (outs
    TFL_FpTensor:$y
  );

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_SliceOp : TFL_Op<"slice", [
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    SameOperandsAndResultsScale,
    TFL_OperandHasRankAtMost<0, 5>,
    TFL_OperandHasRankAtMost<1, 1>,
    TFL_OperandHasRankAtMost<2, 1>]> {
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
    TFL_TensorOf<[F32, I32, I64, I8, UI8, I1, TFL_Str, QI8, QUI8, TFL_Quint8, QI16]>:$input,
    TFL_I32OrI64Tensor:$begin,
    TFL_I32OrI64Tensor:$size
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, I8, UI8, I1, TFL_Str, QI8, QUI8, TFL_Quint8, QI16]>:$output
  );

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TFL_SumOp: TFL_Op<"sum", [
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    // TODO(b/215655380): Re-enable this once there is 16-bit MLIR quantizer.
    // SameOperandsAndResultsScale,
    Pure]> {

  let summary = "Sum operator";

  let description = [{
    Computes the sum reduction along the specified axes
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$input,
    TFL_I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$output);

  let hasOptions = 1;
  let customOption = "ReducerOptions";

  // TODO(b/215655380): Re-enable this once there is 16-bit MLIR quantizer.
  //
  //let extraClassDeclaration = [{
  //  // SameScalesOpInterface:
  //  bool RequiredSameOperandsAndResultsScale(bool sign, int bit_width) {
  //    // Eight-bit types don't require same operands and results scales.
  //    return bit_width != 8;
  //  }
  //}];
}

def TFL_ReduceMinOp: TFL_Op<"reduce_min", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale]> {
  let summary = "Min-reduction operator";

  let description = [{
    Computes the min reduction along the specified axes
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$input,
    TFL_I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$output);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_ReduceMaxOp: TFL_Op<"reduce_max", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale]> {
  let summary = "Max-reduction operator";

  let description = [{
    Computes the max reduction along the specified axes
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$input,
    TFL_I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$output);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_ReduceProdOp: TFL_Op<"reduce_prod", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    QuantizableResult,
    Pure]> {
  let summary = "Prod-reduction operator";

  let description = [{
    Computes the product along the specified axes
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$input,
    TFL_I32Tensor:$axes,
    BoolAttr:$keep_dims
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$output);

  let hasOptions = 1;
  let customOption = "ReducerOptions";
}

def TFL_MinimumOp : TFL_Op<"minimum", [
    ResultsBroadcastableShape,
    Pure,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 5>,
    Commutative,
    QuantizableResult,
    SameOperandsAndResultsScale]> {
  let summary = "Min operator";
  let description = [{
    Element-wise min operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, TFL_Int32Or64, QI8, QUI8, QI16]>:$lhs,
    TFL_TensorOf<[F32, TFL_Int32Or64, QI8, QUI8, QI16]>:$rhs
  );

  let results = (outs
    TFL_TensorOf<[F32, TFL_Int32Or64, QI8, QUI8, QI16]>:$min
  );

  let builders = [TFL_BroadcastableBinaryBuilder];

  let hasOptions = 0;
}

def TFL_MulOp : TFL_Op<"mul", [
    ResultsBroadcastableShape,
    Pure,
    Commutative,
    QuantizableResult,
    BinaryOpSameElementTypeConstraint,
    TFL_RuntimePredOpTrait<"Operands do not have valid shapes",
      CPred<"TFL::VerifyMulOpShapeConstraints(llvm::cast<MulOp>($_op))">>,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Multiplication operator";

  let description = [{
    Element-wise multiplication operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I32, I64, QI8, QUI8, QI16, Complex<F<32>>]>:$lhs,
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, QI16, Complex<F<32>>]>:$rhs,
    TFL_AFAttr:$fused_activation_function);

  let results = (outs TFL_TensorOf<[F32, I32, I64, QI8, QUI8, QI16, Complex<F<32>>]>:$output);

  let hasFolder = 1;

  let builders = [TFL_FusedBroadcastableBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 1;
}

def TFL_NegOp: TFL_Op<"neg", [
    Pure,
    TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Negation operator";

  let description = [{
    Computes element-wise negation of input
  }];

  let arguments = (ins TFL_TensorOf<[F32, I32, I64]>:$x);

  let results = (outs TFL_TensorOf<[F32, I32, I64]>:$y);

  let hasOptions = 0b1;

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_PackOp : TFL_Op<"pack", [
    TFL_SameFirstOperandAndFirstResultElementType,
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale]> {
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
    TFL_VariadicTensorOf<[F32, I8, I16, I32, I64, UI8, QI8, QUI8, QI16, TFL_Quint8]>:$values,

    ConfinedAttr<I32Attr, [IntPositive]>:$values_count,
    I32Attr:$axis
  );

  let results = (outs
    TFL_TensorOf<[F32, I8, I16, I32, I64, UI8, QI8, QUI8, QI16, TFL_Quint8]>:$output
  );

  let hasVerifier = 1;

  let hasCanonicalizer = 1;

  let hasOptions = 1;
}

def TFL_PadOp : TFL_Op<"pad", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    SameOperandsAndResultsScale,
    TFL_OperandHasRankAtMost<0, 5>,
    TFL_OperandHasRank<1, 2>,
    TFL_OperandRankEquals1DimOfOperand<0, 1>,
    QuantizableResult,
    PredOpTrait<"the first dim size of the padding argument must be at most 5",
      Or<[TFL_OperandIsUnrankedPred<1>,
          TFL_OperandDimIsAtMost<1, 0, 5>]>>]> {
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

  let arguments = (ins TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$input,
    TFL_I32OrI64Tensor:$padding);

  let results = (outs TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8, QI16]>:$output);

  let hasOptions = 1;

  let hasFolder = 1;
}

def TFL_PadV2Op : TFL_Op<"padv2", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale,
    TFL_OperandHasRankAtMost<0, 5>,
    TFL_OperandHasRank<1, 2>,
    TFL_OperandHasRank<2, 0>,
    TFL_OperandRankEquals1DimOfOperand<0, 1>,
    PredOpTrait<"the first dim size of the padding argument must be at most 5",
      Or<[TFL_OperandIsUnrankedPred<1>,
          TFL_OperandDimIsAtMost<1, 0, 5>]>>,
    PredOpTrait<"input and constant value operands must have same element type",
      TFL_TCopVTEtAreSameAt<0, 2>>]> {
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
    ins TFL_TensorOf<[F32, I32, I64, UI8, QI8, QUI8, TFL_Quint8]>:$input,
    TFL_I32OrI64Tensor:$padding,
    TFL_TensorOf<[F32, I32, I64, UI8, QI8, QUI8, TFL_Quint8]>:$constant_values);

  let results = (outs TFL_TensorOf<[F32, I32, I64, UI8, QI8, QUI8, TFL_Quint8]>:$output);

  let hasOptions = 1;

  let hasFolder = 1;
}

def TFL_PolyCallOp : Op<TFL_Dialect, "poly_call", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = [{Poly call}];

  let description = [{
    Have multiple function bodies for the same computation. This allows a
    program compiler/interpreter to choose one of the available options to
    execute the program based on which one is most suitable for the target
    backend.

    input:  A list of input tensors whose types are T.
    output: A list of output tensors whose types are T.

    call:  Multiple regions, each of which encapsulates the same semantic
           computation but in different forms.
  }];

  let arguments = (ins Variadic<AnyTensor>:$input);

  let results = (outs Variadic<AnyTensor>:$output);

  let regions = (region VariadicRegion<SizedRegion<1>>:$calls);

  let hasCanonicalizer = 1;
}


def TFL_PowOp : TFL_Op<"pow", [
    ResultsBroadcastableShape,
    Pure,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>]> {
  let summary = "Power operator";

  let description = [{
    Element-wise power operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I32]>:$lhs,
    TFL_TensorOf<[F32, I32]>:$rhs);

  let results = (outs TFL_TensorOf<[F32, I32]>:$output);

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let builders = [TFL_BroadcastableBinaryBuilder];
}

def TFL_PReluOp : TFL_Op<"prelu", [
    Pure,
    QuantizableResult,
    ResultsBroadcastableShape,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    BinaryOpSameElementTypeConstraint,
    PredOpTrait<"input and output must have the same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    AffineQuantizedOpInterface, AffineOpCoefficient<-1, 1>]> {
  let summary = "Parameterized Relu operator";

  let description = [{
    Parameterized Relu operator
      x -> x >= 0 ? x : (alpha * x)
    where alpha is a trainable tensor.
    input and alpha should be the same size as input or be broadcastable.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, QI8, QUI8, TFL_Quint8]>:$input,
    TFL_TensorOf<[F32, QI8, QUI8, TFL_Quint8]>:$alpha
  );

  let results = (outs TFL_TensorOf<[F32, QI8, QUI8, TFL_Quint8]>:$output);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // AffineQuantizedOpInterface:
    int GetChannelDimIndex() { return 0; }
    int GetQuantizationDimIndex() { return -1; }
  }];
}

def TFL_RankOp: TFL_Op<"rank", [
    QuantizableResult,
    Pure]> {
  let summary = "Rank operator.";
  let description = [{
    Returns the rank of a tensor.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs TFL_IntTensor:$output);

  let hasFolder = 1;
}

def TFL_ReluOp: TFL_Op<"relu", [
    PredOpTrait<"x and y must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    QuantizableResult,
    SameOperandsAndResultShape]> {
  let summary = "Relu operator";

  let description = [{
    Element-wise Relu operator
      x -> max(0, x)
  }];

  let arguments = (ins TFL_TensorOf<[F32, QUI8, QI8, QI16]>:$x);

  let results = (outs TFL_TensorOf<[F32, QUI8, QI8, QI16]>:$y);

  // This builder doesn't work with quantized type, so it can only be used by
  // non-quantization tablegen patterns. Currently, it is used by the
  // elementwise-move reordering pattern in the optimize_patterns.td
  let builders = [
    OpBuilder<(ins "Value":$input),
    [{
      $_state.addOperands({input});
      $_state.addTypes(input.getType());
    }]>
  ];
}

def TFL_Relu6Op: TFL_Op<"relu6", [
    PredOpTrait<"x and y must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    QuantizableResult,
    SameOperandsAndResultShape]> {
  let summary = "Relu6 operator";

  let description = [{
    Element-wise Relu6 operator
      x -> max(0, min(6, x))
  }];

  let arguments = (ins TFL_TensorOf<[F32, QUI8, QI8]>:$x);

  let results = (outs TFL_TensorOf<[F32, QUI8, QI8]>:$y);

  // This builder doesn't work with quantized type, so it can only be used by
  // non-quantization tablegen patterns. Currently, it is used by the
  // elementwise-move reordering pattern in the optimize_patterns.td
  let builders = [
    OpBuilder<(ins "Value":$input),
    [{
      $_state.addOperands({input});
      $_state.addTypes(input.getType());
    }]>
  ];
}

def TFL_Relu0To1Op: TFL_Op<"relu_0_to_1", [
    PredOpTrait<"x and y must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    QuantizableResult,
    SameOperandsAndResultShape]> {
  let summary = "Relu0To1 operator";

  let description = [{
    Element-wise Relu0To1 operator
      x -> max(0, min(1, x))
  }];

  let arguments = (ins TFL_TensorOf<[F32, QUI8, QI8]>:$x);

  let results = (outs TFL_TensorOf<[F32, QUI8, QI8]>:$y);
}

def TFL_Relu1Op: TFL_Op<"relu_n1_to_1", [
    PredOpTrait<"x and y must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    QuantizableResult,
    SameOperandsAndResultShape]> {
  let summary = "Relu1 operator";

  let description = [{
    Element-wise Relu1 operator
      x -> max(-1, min(1, x))
  }];

  let arguments = (ins TFL_TensorOf<[F32, QUI8, QI8]>:$x);

  let results = (outs TFL_TensorOf<[F32, QUI8, QI8]>:$y);

  // This builder doesn't work with quantized type, so it can only be used by
  // non-quantization tablegen patterns. Currently, it is used by the
  // elementwise-move reordering pattern in the optimize_patterns.td
  let builders = [
    OpBuilder<(ins "Value":$input),
    [{
      $_state.addOperands({input});
      $_state.addTypes(input.getType());
    }]>
  ];
}

def TFL_ReshapeOp: TFL_Op<"reshape", [
    QuantizableResult,
    Pure,
    SameOperandsAndResultsScale,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Reshape operator";

  let description = [{
    Produces a tensor with the same values but different static shape defined
    by the output type.
  }];

  let arguments = (
    ins AnyTensor:$input,
    TFL_I32Tensor:$shape);

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 0b1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

def TFL_ReverseSequenceOp : TFL_Op<"reverse_sequence", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    QuantizableResult,
    TFL_OperandHasRank<1, 1>]> {
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
    TFL_TensorOf<[F32, I32, I64, QI16, QUI8, TFL_Quint8]>:$input,
    TFL_I32OrI64Tensor:$seq_lengths,

    ConfinedAttr<I32Attr, [IntNonNegative]>:$seq_dim,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$batch_dim
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, QI16, QUI8, TFL_Quint8]>:$output
  );

  let hasOptions = 1;
}

def TFL_RsqrtOp: TFL_Op<"rsqrt", [Pure,
                                  QuantizableResult,
                                  TFL_SameFirstOperandAndFirstResultElementType,
                                  SameOperandsAndResultShape]> {
  let summary = "Reciprocal of square root operator";

  let description = [{
    Computes element-wise reverse square root of input
  }];

  let arguments = (ins TFL_TensorOf<[F32, QI8, QI16]>:$x);

  let results = (outs TFL_TensorOf<[F32, QI8, QI16]>:$y);

  let hasFolder = 1;
}

def TFL_ShapeOp: TFL_Op<"shape", [
    QuantizableResult,
    Pure]> {
  let summary = "Shape operator";

  let description = [{
    Returns the shape of a tensor.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs TFL_TensorOf<[I32, I64]>:$output);

  DerivedTypeAttr out_type = DerivedTypeAttr<[{
    return getResult().getType().cast<TensorType>().getElementType();
  }]>;

  let hasOptions = 1;

  let hasFolder = 1;
}

def TFL_RangeOp: TFL_Op<"range", [
    Pure,
    TFL_OperandHasRank<0, 0>,
    TFL_OperandHasRank<1, 0>,
    TFL_OperandHasRank<2, 0>,
    PredOpTrait<"operands and output must have same element type",
      And<[TCresVTEtIsSameAsOp<0, 0>, TCresVTEtIsSameAsOp<0, 1>,
           TCresVTEtIsSameAsOp<0, 2>]>>]> {
  let summary = "Range operator";

  let description = [{
    Returns a 1D tensor defined by a sequence from `start` to `limit` with
    a given `delta`.
  }];

  let arguments = (ins
    TFL_TensorOf<[I32, F32]>:$start,
    TFL_TensorOf<[I32, F32]>:$limit,
    TFL_TensorOf<[I32, F32]>:$delta);

  let results = (outs TFL_TensorOf<[I32, F32]>:$result);

  let hasFolder = 1;
}

def TFL_ReverseV2Op: TFL_Op<"reverse_v2", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    Pure,
    QuantizableResult,
    TFL_OperandHasRank<1, 1>]> {
  let summary = "ReverseV2 Operator";

  let description = [{
    Reverses specific dimensions of a tensor.

    Given a tensor, and a int32/int64 tensor axis representing the set
    of dimensions of tensor to reverse.
    This operation reverses each dimension i for
    which there exists j s.t. axis[j] == i.

    Args:
      tensor: A Tensor. Must be one of the following types:
      uint8, int8, int16, int32, int64, float32, bool Up to 8-D.

      axis: A Tensor. Must be one of the following types: int32, int64.
      with only 1 element which is the axis index.
      TODO: Add support for multiple elements.
  }];

  let arguments = (
    ins
    TFL_TensorOf<[F32, UI8, I16, I32, I64, QI16, QUI8, QI8, TFL_Quint8, I1]>:$input,
    TFL_I32Tensor:$axis
  );

  let results = (outs
    TFL_TensorOf<[F32, UI8, I16, I32, I64, QI16, QUI8, QI8, TFL_Quint8, I1]>:$output);
}

// Select has many instances in TF models where one or more of its operands
// are unranked. Therefore, we skip adding shape constraints here.
def TFL_SelectOp : TFL_Op<"select", [
  Pure,
  SameOperandsAndResultsScale,
  QuantizableResult,
  PredOpTrait<"operands have same element type", TFL_TCopVTEtAreSameAt<1, 2>>,
  PredOpTrait<"operands and result have same element type",
    TFL_TCresVTEtIsSameAsOp<0, 1>>]> {
  let summary = "Select operator";

  let description = [{
    Select values of 'x' if the corresponding value of 'condition' is true or
    the value of 'y' if false. There are valid condition input sizes:

    1. Either the same shape (in which case the select is elementwise), or
    2. condition must be Rank 1 and match over the first dimension.
  }];

  let arguments = (ins
    TFL_BoolTensor:$condition,
    TFL_TensorOf<[F32, I1, I8, I16, I32, I64, QI8, QUI8, QI16, TFL_Quint8]>:$x,
    TFL_TensorOf<[F32, I1, I8, I16, I32, I64, QI8, QUI8, QI16, TFL_Quint8]>:$y);

  let results = (outs
    TFL_TensorOf<[F32, I1, I8, I16, I32, I64, QI8, QUI8, QI16, TFL_Quint8]>:$output);

  // TODO(jpienaar): autogenerate this.
  let builders = [
    OpBuilder<(ins "Value":$condition, "Value":$x, "Value":$y),
    [{
    auto resultType = x.getType();
    $_state.addOperands({condition, x, y});
    $_state.types.push_back(resultType);
  }]>];

  let hasOptions = 1;
}

def TFL_SelectV2Op : TFL_Op<"select_v2", [
    ResultsBroadcastableShape,
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale,
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1, 2], 5>,
    PredOpTrait<"operands have same element type", TFL_TCopVTEtAreSameAt<1, 2>>,
    PredOpTrait<"operands and result have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 1>>]> {
  let summary = "SelectV2 operator";

  let description = [{
    Select values of 'x' if the corresponding value of 'condition' is true or
    the value of 'y' if false. There are valid condition input sizes:

    1. Either the same shape (in which case the select is elementwise), or
    2. Broadcastable shapes between 'condition', 'x' and 'y'.
  }];

  let arguments = (ins
    TFL_BoolTensor:$condition,
    TFL_TensorOf<[F32, I1, I8, I16, I32, I64, QI8, QUI8, QI16, TFL_Quint8]>:$x,
    TFL_TensorOf<[F32, I1, I8, I16, I32, I64, QI8, QUI8, QI16, TFL_Quint8]>:$y);

  let results = (outs
    TFL_TensorOf<[F32, I1, I8, I16, I32, I64, QI8, QUI8, QI16, TFL_Quint8]>:$output);

  let builders = [
    OpBuilder<(ins "Value":$cond, "Value":$x, "Value":$y),
    [{
    BuildSelectV2Op(&$_builder, $_state, cond, x, y);
  }]>];

  let hasOptions = 1;
}

def TFL_SinOp: TFL_Op<"sin", [
    Pure,
    TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Sine operator";

  let description = [{
    Computes element-wise Sine of input
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_SoftmaxOp : TFL_Op<"softmax", [
    Pure,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandHasRankAtLeast<0, 1>,
    SameOperandsAndResultShape,
    QuantizableResult,
    FixedOutputRangeInterface,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Softmax operator";

  let description = [{
    Computes element-wise softmax activations with the following formula

      exp(input) / tf.reduce_sum(exp(input * beta), dim)
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, QI8, QUI8, TFL_Quint8, QI16]>:$input,
    F32Attr:$beta
  );

  let results = (outs TFL_TensorOf<[F32, QI8, QUI8, TFL_Quint8, QI16]>:$output);

  let hasOptions = 1;

  let extraClassDeclaration = [{
  // FixedOutputRangeInterface:
  quant::UniformQuantizedType GetFixedOutputRange(
      bool is_signed, int bit_width) {
    auto result_type = getOutput().getType();
    // zero_point = 0
    // scale = 1. / (max_value + 1)
    return quant::GetFixedOutputRange(is_signed, bit_width, result_type,
        /*scale=*/1.0 / (1<<(bit_width)),
        /*zero_point=*/-(1<<(bit_width-1)));
  }
  }];
}

def TFL_SqrtOp: TFL_Op<"sqrt", [
    Pure,
    TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Square root operator";

  let description = [{
    Computes element-wise Square root of input
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_SquareOp: TFL_Op<"square", [
    Pure,
    TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Square operator";

  let description = [{
    Computes element-wise Square of input
  }];

  let arguments = (ins TFL_FpTensor:$x);

  let results = (outs TFL_FpTensor:$y);

  let hasOptions = 0b1;

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return TF::ArraysAreCastCompatible(l, r);
    }
  }];
}

def TFL_SubOp : TFL_Op<"sub", [
    ResultsBroadcastableShape,
    BinaryOpSameElementTypeConstraint,
    TFL_RuntimePredOpTrait<"Operands do not have valid shapes",
      CPred<"TFL::VerifySubOpShapeConstraints(llvm::cast<SubOp>($_op))">>,
    Pure,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Subtraction operator";

  let description = [{
    Element-wise subtraction operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I32, I64, QI8, QUI8, QI16]>:$lhs,
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, QI16]>:$rhs,
    TFL_AFAttr:$fused_activation_function);

  let results = (outs TFL_TensorOf<[F32, I32, I64, QI8, QUI8, QI16]>:$output);

  let hasFolder = 1;

  let builders = [TFL_FusedBroadcastableBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];

  let hasOptions = 1;
}

def TFL_SquaredDifferenceOp : TFL_Op<"squared_difference", [
    TFL_OperandsHaveSameShapesOrBroadcastableShape<[0, 1], 4>,
    BinaryOpSameElementTypeConstraint,
    TFL_SameFirstOperandAndFirstResultElementType,
    ResultsBroadcastableShape,
    QuantizableResult,
    Pure]> {
  let summary = "Squared difference operator";

  let description = [{
    Element-wise squared difference operation.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I32, QI8]>:$lhs,
    TFL_TensorOf<[F32, I32, QI8]>:$rhs);

  let results = (outs TFL_TensorOf<[F32, I32, QI8]>:$output);

  let builders = [TFL_BroadcastableBinaryBuilder];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return parseOneResultSameOperandTypeOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      return printOneResultOp(getOperation(), p);
    }
  }];
}

def TFL_TanhOp: TFL_Op<"tanh", [
    Pure,
    SameOperandsAndResultShape,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    FixedOutputRangeInterface,
    QuantizableResult,
    DeclareOpInterfaceMethods<TFL_ArithmeticCount>]> {
  let summary = "Hyperbolic tangent operator";

  let description = [{
    Computes element-wise Hyperbolic tangent of input
  }];

  let arguments = (ins TFL_TensorOf<[F32, QI8, QUI8, QI16, TFL_Quint8]>:$input);

  let results = (outs TFL_TensorOf<[F32, QI8, QUI8, QI16, TFL_Quint8]>:$output);

  // This builder doesn't work with quantized type, so it can only be used by
  // non-quantization tablegen patterns. Currently, it is used by the
  // elementwise-move reordering pattern in the optimize_patterns.td
  let builders = [
    OpBuilder<(ins "Value":$input),
    [{
      $_state.addOperands({input});
      $_state.addTypes(input.getType());
    }]>
  ];

  let extraClassDeclaration = [{
  // FixedOutputRangeInterface:
  quant::UniformQuantizedType GetFixedOutputRange(
      bool is_signed, int bit_width) {
    auto result_type = getOutput().getType();
    // central_value = min_value / 2 + (max_value - 1) / 2 + 1
    // zero_point = central_value
    // scale = 1. / (central_value - min_value)
    return quant::GetFixedOutputRange(is_signed, bit_width, result_type,
        /*scale=*/1.0 / (1<<(bit_width-1)), /*zero_point=*/0);
  }
  }];
}

def TFL_TileOp: TFL_Op<"tile", [
    Pure,
    SameOperandsAndResultsScale,
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
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
    TFL_TensorOf<[F32, I1, I32, I64, UI8, QI8, QUI8, TFL_Str]>:$input,
    TFL_I32OrI64Tensor:$multiples);

  let results = (outs
    TFL_TensorOf<[F32, I1, I32, I64, UI8, QI8, QUI8, TFL_Str]>:$output);

  let hasOptions = 0;
}

// TODO(jpienaar): Maybe make it accept any single element tensor as `k`.
// TODO(jpienaar): Check that input has one or more dimensions.
// TODO(jpienaar): Check that k is less or equal the internal dimension
def TFL_TopKV2Op: TFL_Op<"topk_v2", [
    QuantizableResult,
    Pure,
    TFL_OperandHasRankAtLeast<0, 1>,
    TFL_OperandHasRank<1, 0>,
    PredOpTrait<"result and input element type match",
      TFL_TCresVTEtIsSameAsOp<0,0>>,
    SameOperandsAndResultsScale]> {
  let summary = "TopK operator";

  let description = [{
    Returns the top `k` largest element along each last dimensional slice of
    `input` and the indices of values within the last dimension of the input
    tensor.

    Results are always sorted in the descending order.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I8, I32, I64, UI8, QI8, QUI8]>:$input,
    TFL_I32Tensor:$k);

  let results = (outs
    TFL_TensorOf<[F32, I8, I32, I64, UI8, QI8, QUI8]>:$values,
    TFL_I32Tensor:$indices);

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$k),
    [{ BuildTopKOp(&$_builder, $_state, input, k); }]>];

  let hasOptions = 1;
}

def TFL_TransposeOp : TFL_Op<"transpose", [
    Pure,
    QuantizableResult,
    TFL_OperandHasRankAtMost<0, 6>,
    TFL_OperandHasRank<1, 1>,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    SameOperandsAndResultsScale]> {
  let summary = "Transpose operator";

  let description = [{
    Returns the Transpose of x
  }];

  let arguments = (ins
    TFL_TensorOf<[I32, F32, I8, UI8, QI8, QUI8, TFL_Quint8, I1, I64, QI16]>:$input,
    TFL_TensorOf<[I32]>:$perm
  );

  let results = (outs
    TFL_TensorOf<[I32, F32, I8, UI8, QI8, QUI8, TFL_Quint8, I1, I64, QI16]>:$output
  );

  let hasVerifier = 1;

  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$perm),
    [{ BuildTransposeOp(&$_builder, $_state, input, perm); }]>
  ];

  let extraClassDeclaration = [{
    // Quantized axes are verified in the Verify function.
    bool RequiredSameQuantizedAxes() { return false; }
  }];
}

def TFL_UnpackOp : TFL_Op<"unpack", [
    Pure,
    QuantizableResult,
    SameOperandsAndResultElementType,
    SameOperandsAndResultsScale,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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
    TFL_TensorOf<[F32, I1, I8, UI8, I32, QI8, QUI8, I16, QI16]>:$input,

    ConfinedAttr<I32Attr, [IntNonNegative]>:$num,
    I32Attr:$axis
  );

  let results = (outs
    TFL_VariadicTensorOf<[F32, I1, I8, UI8, I32, QI8, QUI8, I16, QI16]>:$outputs
  );

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];

  let hasOptions = 1;
}

def TFL_ZerosLikeOp: TFL_Op<"zeros_like", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    SameOperandsAndResultShape,
    Pure]> {
  let summary = "ZerosLike operator";

  let description = [{
    Returns a tensor of zeros with the same shape and type as the input tensor.
  }];

  let arguments = (ins TFL_TensorOf<[I64, I32, F32]>:$input);

  let results = (outs TFL_TensorOf<[I64, I32, F32]>:$output);

  let hasOptions = 1;
}

def TFL_BatchToSpaceNdOp: TFL_Op<"batch_to_space_nd", [
    Pure,
    SameOperandsAndResultsScale,
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandHasRankRange<0, 3, 4>,
    TFL_OperandHasRank<1, 1>,
    TFL_OperandHasRank<2, 2>
  ]> {
  let summary = "BatchToSpaceNd operator";

  let description = [{
    This operation reshapes the "batch" dimension 0 into space dimensions.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I8, I32, I64, UI8, QI8, QUI8]>:$input,
    TFL_TensorOf<[I32]>:$block_shape,
    TFL_TensorOf<[I32]>:$indices
  );

  let results = (outs
    TFL_TensorOf<[F32, I16, I32, I64, UI8, QI8, QUI8]>:$output
  );
}

def TFL_SpaceToBatchNdOp: TFL_Op<"space_to_batch_nd", [
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale,
    TFL_OperandHasRankRange<0, 3, 4>,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>
  ]> {
  let summary = "SpaceToBatchNd operator";

  let description = [{
    This operation reshapes space dimensions into the "batch" dimension 0
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8]>:$input,
    TFL_I32Tensor:$block_shape,
    TFL_I32Tensor:$paddings
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8]>:$output
  );
}

def TFL_SpaceToDepthOp: TFL_Op<"space_to_depth", [
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandHasRankAtMost<0, 4>
  ]> {
  let summary = "SpaceToDepth operator";

  let description = [{
    Rearranges blocks of spatial data, into depth. More specifically,
    this op outputs a copy of the input tensor where values from the `height`
    and `width` dimensions are moved to the `depth` dimension.
    `block_size` indicates the input block size.
   }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8]>:$input,
    ConfinedAttr<I32Attr, [IntPositive]>:$block_size
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, QI8, QUI8, TFL_Quint8]>:$output
  );

  let hasOptions = 1;
}

def TFL_DepthToSpaceOp: TFL_Op<"depth_to_space", [
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandHasRankAtMost<0, 4>
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
    TFL_TensorOf<[F32, I8, I32, I64, TFL_Quint8, UI8, QI8, QUI8]>:$input,
    ConfinedAttr<I32Attr, [IntPositive]>:$block_size
  );

  let results = (outs
    TFL_TensorOf<[F32, I8, I32, I64, TFL_Quint8, UI8, QI8, QUI8]>:$output
  );

  let hasOptions = 1;
}

def TFL_SplitOp : TFL_Op<"split", [
    Pure,
    TFL_Operand0DOr1ElementTensor<0>,
    QuantizableResult,
    SameOperandsAndResultsScale]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let description = [{
    Splits the `value` tensor along `split_dim` into a number of sub-tensors
    with same shape as the original one, except for `split_dim`. Same as
    tf.Split.
  }];

  let arguments = (ins
    TFL_TensorOf<[I32]>:$split_dim,
    TFL_TensorOf<[F32, I16, I32, I8, UI8, QI8, QUI8, QI16]>:$value,
    ConfinedAttr<I32Attr, [IntPositive]>:$num_splits
  );

  let results = (outs
    TFL_VariadicTensorOf<[F32, I16, I32, I8, UI8, QI8, QUI8, QI16]>:$outputs
  );

  let hasVerifier = 1;

  let hasOptions = 1;
}

def TFL_SplitVOp : TFL_Op<"split_v", [
    Pure,
    QuantizableResult,
    SameOperandsAndResultsScale]> {
  let summary = "Splits a tensor into `num_split` tensors along one dimension.";

  let description = [{
    Splits the `value` tensor along `split_dim` into a number of sub-tensors
    with same shape as the original one, except for `split_dim`. The grouping
    of the resultant sub-tensors is decided by `size-splits`. Same as tf.SplitV.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I16, I32, I64, I8, UI8, QI8, QUI8, QI16]>:$value,
    TFL_1DTensorOf<[I32], [I32]>:$size_splits,
    TFL_0DTensorOf<[I32], [I32]>:$split_dim,
    ConfinedAttr<I32Attr, [IntPositive]>:$num_splits
  );

  let results = (outs
    TFL_VariadicTensorOf<[F32, I16, I32, I64, I8, UI8, QI8, QUI8, QI16]>:$outputs
  );

  let hasVerifier = 1;

  let hasOptions = 1;
}

def TFL_ResizeBilinearOp: TFL_Op<"resize_bilinear", [
    Pure,
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandHasRank<0, 4>,
    TFL_OperandHasRank<1, 1>,
    SameOperandsAndResultsScale]> {
  let summary = "ResizeBilinear Op";

  let description = [{
    Resize `images` to `size` using bilinear interpolation.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, TFL_Quint8, QUI8, QI8, QI16]>:$input,
    TFL_I32Tensor:$size,
    BoolAttr:$align_corners,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    TFL_TensorOf<[F32, TFL_Quint8, QUI8, QI8, QI16]>:$output
  );

  let hasOptions = 1;
}

def TFL_ResizeNearestNeighborOp : TFL_Op<"resize_nearest_neighbor", [
    Pure,
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandHasRank<0, 4>,
    TFL_OperandHasRank<1, 1>,
    SameOperandsAndResultsScale]> {
  let summary = "ResizeNearestNeighbor Op";

  let description = [{
    Resize `images` to `size` using nearest neighbor interpolation.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, TFL_Quint8, QUI8, QI8, QI16]>:$input,
    TFL_I32Tensor:$size,
    BoolAttr:$align_corners,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$half_pixel_centers
  );

  let results = (outs
    TFL_TensorOf<[F32, TFL_Quint8, QUI8, QI8, QI16]>:$output
  );

  let hasOptions = 1;
}

def TFL_SparseToDenseOp : TFL_Op<"sparse_to_dense", [
    Pure,
    QuantizableResult,
    PredOpTrait<"sparse_values and dense must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 2>>,
    PredOpTrait<"default_value and dense must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 3>>,
    TFL_OperandHasRankAtMost<0, 2>,
    TFL_OperandHasRankAtMost<1, 1>,
    TFL_OperandHasRankAtMost<2, 1>,
    PredOpTrait<"the first operand should have a rank <= 2, when its rank is 2 and has static shape, the second dim should be <= 4",
      Or<[TFL_OperandIsUnrankedPred<0>,
          CPred<"$_op.getOperand(0).getType().cast<ShapedType>().getRank() <= 1">,
          CPred<"$_op.getOperand(0).getType().cast<ShapedType>().getRank() == 2 && !$_op.getOperand(0).getType().cast<ShapedType>().hasStaticShape()">,
          CPred<"$_op.getOperand(0).getType().cast<ShapedType>().getRank() == 2 && $_op.getOperand(0).getType().cast<ShapedType>().getShape()[1] <= 4">]>>]> {
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
    TFL_TensorOf<[I32, I64, I8, QI8, UI8, QUI8, TFL_Quint8, F32]>:$sparse_values,
    TFL_TensorOf<[I32, I64, I8, QI8, UI8, QUI8, TFL_Quint8, F32]>:$default_value
  );

  let results = (outs
    TFL_TensorOf<[I32, I64, I8, QI8, UI8, QUI8, TFL_Quint8, F32]>:$dense
  );
}

def TFL_StridedSliceOp: TFL_Op<"strided_slice", [
    Pure,
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    SameOperandsAndResultsScale,
    TFL_OperandHasRankAtMost<0, 5>,
    TFL_OperandHasRank<1, 1>,
    TFL_OperandHasRank<2, 1>,
    TFL_OperandHasRank<3, 1>
  ]> {
  let summary = "StridedSlice Op";

  let description = [{
    Return a strided slice from `input`.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32, I64, I8, UI8, QI8, QUI8, I1, I16, QI16, TFL_Quint8, TFL_Str]>:$input,
    TFL_I32Tensor:$begin,
    TFL_I32Tensor:$end,
    TFL_I32Tensor:$strides,

    I32Attr:$begin_mask,
    I32Attr:$end_mask,
    I32Attr:$ellipsis_mask,
    I32Attr:$new_axis_mask,
    I32Attr:$shrink_axis_mask
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, I8, UI8, QI8, QUI8, I1, I16, QI16, TFL_Quint8, TFL_Str]>:$output
  );

  // TFLite kernel only supports up to 5D input including added axis.
  let hasVerifier = 1;

  let hasOptions = 1;

  let hasFolder = 1;
}

// If there is a change in supporting more types in the TFLite cast op kernel,
// the While loop outline pass should be updated since it inserts cast op(s)
// after the TF -> TFL legalization pass is done.
// LINT.IfChange
def TFL_CastOp : TFL_Op<"cast", [
    Pure,
    SameOperandsAndResultShape]> {
  let summary = "Cast operator";

  let description = [{
    Casts input from input type to output type.
  }];

  let arguments = (ins
    TFL_TensorOf<[F16, F32, F64, I1, I16, UI16, I32, UI32, I64, TFL_Quint8, UI8, I8, Complex<F<32>>]>:$input
  );

  let results = (outs TFL_TensorOf<[F16, F32, F64, I1, I16, UI16, I32, UI32, I64, TFL_Quint8, UI8, I8, Complex<F<32>>]>:$output);

  // TFLite's cast op does not utilize CastOptions, instead derives types
  // from the TfLiteTensors.
  let hasOptions = 0;

  let hasFolder = 1;
}
// LINT.ThenChange(//tensorflow/compiler/mlir/lite/transforms/while_loop_outline.cc)

def TFL_MirrorPadOp: TFL_Op<"mirror_pad", [
                     SameOperandsAndResultsScale,
                     QuantizableResult,
                     Pure,
                     TFL_OperandHasRank<1, 2>]> {
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
    TFL_TensorOf<[F32, I32, I64, I8, UI8, QI8, QUI8, QI16]>:$input,
    TFL_TensorOf<[I32, I64]>:$pad,
    TFL_MirrorPaddingAttr:$mode
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I64, I8, UI8, QI8, QUI8, QI16]>:$output
  );

  let hasOptions = 1;
}

def TFL_UniqueOp: TFL_Op<"unique", [
    TFL_OperandHasRank<0, 1>,
    QuantizableResult,
    Pure]> {
  let summary = "Unique Op.";

  let description = [{
  This operation returns a tensor `output` containing all of the unique elements
of `input` sorted in the same order that they occur in `input`. This operation
also returns a tensor `idx` the same size as `x` that contains the index of each
value of `input` in the unique output `output`. In other words:
  }];

  let arguments = (ins
    TFL_TensorOf<[I8, QI8, UI8, QUI8, I16, QI16, I32, I64, F32]>:$input
  );

  let results = (outs
    TFL_TensorOf<[I8, QI8, UI8, QUI8, I16, QI16, I32, I64, F32]>:$output,
    TFL_I32OrI64Tensor:$idx
  );

  DerivedTFLiteTypeAttr idx_out_type = DerivedTFLiteTypeAttr<[{
    return getResult(1).getType().cast<TensorType>().getElementType().
        cast<IntegerType>().getWidth() > 32 ? tflite::TensorType_INT64 :
            tflite::TensorType_INT32;
    }], [{
      TypeAttr::get(getResult(1).getType().cast<TensorType>().getElementType())
    }]>;

  let hasOptions = 1;
}

def TFL_GeluOp: TFL_Op<"gelu", [
    Pure,
    SameOperandsAndResultShape,
    QuantizableResult,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "GELU activation function.";
  let description = [{
    Computes GELU activation function element-wise.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, QI8, QUI8]>:$input,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$approximate
  );

  let results = (outs TFL_TensorOf<[F32, QI8, QUI8]>:$output);

  let hasOptions = 1;
}

def TFL_DynamicUpdateSliceOp: TFL_Op<"dynamic_update_slice", [
    Pure,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "DynamicUpdateSlice.";
  let description = [{
    DynamicUpdateSlice op that have the same semantics with XLA
    DynamicUpdateSlice.
    Generates a result which is the value of the input array
    operand, with a slice update overwritten at start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice.
  }];

  let arguments = (ins
    TFL_TensorOf<[I1, I8, I32, I64, F32]>:$operand,
    TFL_TensorOf<[I1, I8, I32, I64, F32]>:$update,
    TFL_I32Tensor:$start_indices
  );

  let results = (
    outs TFL_TensorOf<[I1, I8, I32, I64, F32]>:$output);
}

//===----------------------------------------------------------------------===//
// Quantization ops.
//===----------------------------------------------------------------------===//
def TFL_DequantizeOp: TFL_Op<"dequantize", [NoMemoryEffect]> {
  let summary = "Dequantize operator";

  let description = [{
    Converts quantized array of integers to floating-points according to the
    quantization parameters.
  }];

  let arguments = (ins TFL_TensorOf<[QI4, QI8, QUI8, QI16, F16]>:$input);

  let results = (outs TFL_FpTensor:$output);
}

def TFL_FakeQuantOp : TFL_Op<"fake_quant", [
    Pure,
    QuantizableResult]> {
  let summary = "FakeQuant operator";

  let description = [{
    Fake-quantize the 'inputs' tensor of type float via float scalars min and
    max to 'outputs' tensor of same shape as inputs.
  }];

  let arguments = (
    ins TFL_FpTensor:$input,
    // The expected [min, max] range of values.
    F32Attr:$min,
    F32Attr:$max,

    // The bitwidth of the quantization; between 2 and 16, inclusive.
    ConfinedAttr<I32Attr, [IntMinValue<2>, IntMaxValue<16>]>:$num_bits,
    // Quantization range starts from 0 or 1; starts from 1 if true.
    ConfinedAttr<BoolAttr, [TFL_BoolFalse]>:$narrow_range);

  let results = (outs TFL_FpTensor:$output);

  let hasCanonicalizer = 0b1;

  let hasOptions = 1;
}

// TODO(b/200841823): consider adding TFL_RuntimeVerification trait.
def TFL_QConstOp : Op<TFL_Dialect, "pseudo_qconst", [
    Pure,
    FirstAttrDerivedResultType]> {
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

  let results = (outs TFL_TensorOf<[QUI8, QI8, QI16, QUI16, TFL_Quint8]>:$output);

  let builders = [
    OpBuilder<(ins "TypeAttr":$qtype, "Attribute":$value),
    [{
      $_state.addAttribute("qtype", qtype);
      $_state.addAttribute("value", value);
      $_state.addTypes(qtype.getValue());
    }]>
  ];
}

def TFL_SparseQConstOp : Op<TFL_Dialect, "pseudo_sparse_qconst", [
    Pure,
    FirstAttrDerivedResultType,
    DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {
  let summary = "Sparse quantized constant pseudo op";

  let description = [{
    Represents a sparse quantized constant value in TensorFlow Lite dialect.
    This is not an actual operation and it will be lowered to buffer instead.
    The quantization parameters are stored as a type attribute in this constant.
  }];

  let arguments = (
    ins TensorTypeAttr:$qtype,
    ElementsAttr:$value,
    SparsityParameterAttr:$s_param,
    ElementsAttr:$compressed_data
  );

  let results = (outs TFL_TensorOf<[QUI8, QI8, QI16, QUI16, TFL_Quint8]>:$output);

  let builders = [
    OpBuilder<(ins "TypeAttr":$qtype, "Attribute":$value,
      "SparsityParameterAttr":$s_param, "Attribute":$compressed_data),
    [{
      $_state.addTypes(qtype.getValue());
      $_state.addAttribute("qtype", qtype);
      $_state.addAttribute("value", value);
      $_state.addAttribute("s_param", s_param);
      $_state.addAttribute("compressed_data", compressed_data);
    }]>
  ];
}

def TFL_QuantizeOp: TFL_Op<"quantize", [
    FirstAttrDerivedResultType,
    SameOperandsAndResultShape, NoMemoryEffect]> {
  let summary = "Quantize operator";

  let description = [{
    Converts floating point tensors to quantized integer tensors according to
    the quantization parameters defined in the type attribute.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, QI4, QI8, QUI8, QI16, TFL_Quint8]>:$input,
    TensorTypeAttr:$qtype
  );

  let results = (outs TFL_TensorOf<[QI4, QI8, QUI8, QI16, TFL_Quint8]>:$output);
}

def TFL_DensifyOp: TFL_Op<"densify", [
    Pure,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "Densify operator";

  let description = [{
    Converts sparse tensor to dense format.
  }];

  let arguments = (ins TFL_TensorOf<[F32, I8]>:$input);

  let results = (outs TFL_TensorOf<[F32, I8]>:$output);
}

//===----------------------------------------------------------------------===//
// LSTM Ops
//===----------------------------------------------------------------------===//

def LstmMandatoryInputsConstraint : PredOpTrait<
  "mandatory operands element types should match",
  // TODO(ashwinm): Replace the indices with input tensor names when that
  // support is available.
  Or<[
    TCopVTEtAreSameAt<[0, 2, 3, 4, 6, 7, 8, 13, 14, 15, 18, 19]>,
    Neg<TypeIsPred<"input", F32>>]>>;

def LstmOptionalPeepholeWeightConstraint : PredOpTrait<
  "the optional peephole weights should all be specified or none",
  // Ignore input 9 (cell_to_input_weights) for LSTM with CIFG.
  And<[
    TFL_TCopVTEtAreSameAt<10, 11, 16>,
    Or<[TFL_TCopVTEtAreSameAt<9, 10, 16>,
        And<[TypeIsPred<"input_to_input_weights", NoneType>,
             TypeIsPred<"cell_to_input_weights", NoneType>]>]>]>>;

def LstmProjectionWeightBiasConstraint : PredOpTrait<
  "either projection weight must be specified or both projection weight and "
  "projection bias must not be specified",
   Or<[
      And<[TypeIsPred<"projection_weights", NoneType>,
           TypeIsPred<"projection_bias", NoneType>]>,
      Neg<TypeIsPred<"projection_weights", NoneType>>]>>;

def LstmCifgInputConstraint : PredOpTrait<
  "the cifg inputs should all be specified or none",
   // If LSTM has combined input/forget gate, input 1, 5, 9, 12, 20 are all none
   // or 1, 5, 12 should not be none. Inputs 9 and 20 depend on LSTM's variants.
   Or<[
     And<[TypeIsPred<"input_to_input_weights", NoneType>,
          TypeIsPred<"recurrent_to_input_weights", NoneType>,
          TypeIsPred<"cell_to_input_weights", NoneType>,
          TypeIsPred<"input_gate_bias", NoneType>,
          TypeIsPred<"input_layer_norm_coefficients", NoneType>]>,
     Neg<Or<[
       TypeIsPred<"input_to_input_weights", NoneType>,
       TypeIsPred<"recurrent_to_input_weights", NoneType>,
       TypeIsPred<"input_gate_bias", NoneType>]>>]>>;


// TODO(b/137798843): Need to add an additional constraint for both LSTM and
// UnidirectionalSequenceLstm
// For layer norm: if layer norm is false, tensor {20, 21, 22, 23}
// are null; if layer norm is true, tensors {21, 22, 23} are not null; tensor
// {20} is not null if additionally cifg = false.

def LstmResultConstraint : PredOpTrait<
  "the input and result tensor elemental types must be same",
  TFL_TCresVTEtIsSameAsOp<0, 0>>;

// This is the basic kernel type LSTM op.
// TODO(b/142417845): Refactor this part to return its tflite node name as
// "lstm".
def TFL_BasicLSTMOp : TFL_Op<"basic_lstm", [Pure,
    TFL_OperandHasRank<0, 2>,
    TFL_OperandHasRank<1, 2>,
    TFL_OperandHasRank<2, 2>,
    TFL_OperandHasRank<3, 1>,
    TFL_OperandHasRank<4, 2>,
    QuantizableResult]> {
  let summary = "The basic lstm operator";

  let description = [{
    basic LSTM Cell Operator.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, QUI8]>:$data_input,
    TFL_TensorOf<[F32, QUI8]>:$prev_activ_input,
    TFL_TensorOf<[F32, QUI8]>:$weights_input,
    TFL_TensorOf<[F32, QI32]>:$biases_input,
    TFL_TensorOf<[F32, QI16]>:$prev_state_input,

    // Attributes
    DefaultValuedStrAttr<TFL_AFAttr, "TANH">:$fused_activation_function,
    ConfinedAttr<DefaultValuedOptionalAttr<F32Attr, "0.0f">, [TFL_FloatNonNegative]>:$cell_clip,
    ConfinedAttr<DefaultValuedOptionalAttr<F32Attr, "0.0f">, [TFL_FloatNonNegative]>:$proj_clip,
    // Since this op is the BASIC kernel only, constrain it.
    ConfinedAttr<
      DefaultValuedOptionalAttr<TFL_LSTMKernelTypeAttr, "mlir::TFL::LSTMKernelType::BASIC">,
      [TFL_LSTMKernelTypeEqualsTo<"mlir::TFL::LSTMKernelType::BASIC">]>:$kernel_type
  );

  let hasOptions = 1;

  let results = (outs TFL_2DTensorOf<[F32, QUI8]>:$activ_output,
                      TFL_2DTensorOf<[F32, QUI16]>:$state_output,
                      TFL_2DTensorOf<[F32, QUI8]>:$concat_temp,
                      TFL_2DTensorOf<[F32, QUI16]>:$activ_temp);
}

// This is the FULL kernel type LSTM op.
def TFL_LSTMOp :
  TFL_Op<"lstm",
          [LstmMandatoryInputsConstraint,
           LstmOptionalPeepholeWeightConstraint,
           LstmProjectionWeightBiasConstraint,
           LstmCifgInputConstraint,
           LstmResultConstraint,
           TFL_OperandHasRank<2, 2>,           // input_to_forget_weights
           TFL_OperandHasRank<3, 2>,           // input_to_cell_weights
           TFL_OperandIsNoneOrHasRank<5, 2>,   // recurrent_to_input_weights
           TFL_OperandHasRank<6, 2>,           // recurrent_to_forget_weights
           TFL_OperandHasRank<7, 2>,           // recurrent_to_cell_weights
           TFL_OperandIsNoneOrHasRank<9, 1>,   // cell_to_input_weights
           TFL_OperandIsNoneOrHasRank<10, 1>,  // cell_to_forget_weights
           TFL_OperandIsNoneOrHasRank<11, 1>,  // cell_to_output_weights
           TFL_OperandHasRank<13, 1>,          // forget_gate_bias
           TFL_OperandHasRank<14, 1>,          // cell_gate_bias
           TFL_OperandHasRank<15, 1>,          // output_gate_bias
           TFL_OperandIsNoneOrHasRank<16, 2>,  // projection_weights
           TFL_OperandIsNoneOrHasRank<17, 1>,  // projection_bias
           TFL_StatefulOp,
           QuantizableResult,
           DynamicRangeQuantizedOpInterface]> {
  let summary = "The full lstm operator";

  let description = [{
Long short-term memory unit (LSTM) recurrent network layer.
The default non-peephole implementation is based on:
http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
S. Hochreiter and J. Schmidhuber. 'Long Short-Term Memory'. Neural Computation,
9(8):1735-1780, 1997.
The peephole implementation is based on:
https://research.google.com/pubs/archive/43905.pdf
Hasim Sak, Andrew Senior, and Francoise Beaufays. 'Long short-term memory
recurrent neural network architectures for large scale acoustic modeling.'
INTERSPEECH, 2014.
The coupling of input and forget gate (CIFG) is based on:
http://arxiv.org/pdf/1503.04069.pdf
Greff et al. 'LSTM: A Search Space Odyssey'
The layer normalization is based on:
https://arxiv.org/pdf/1607.06450.pdf
Ba et al. 'Layer Normalization'
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, QI8, QI16]>:$input,

    // Weights
    TFL_TensorOfOrNone<[F32, QI8]>:$input_to_input_weights,
    TFL_TensorOf<[F32, QI8]>:$input_to_forget_weights,
    TFL_TensorOf<[F32, QI8]>:$input_to_cell_weights,
    TFL_TensorOf<[F32, QI8]>:$input_to_output_weights,

    // Recurrent weights
    TFL_TensorOfOrNone<[F32, QI8]>:$recurrent_to_input_weights,
    TFL_TensorOf<[F32, QI8]>:$recurrent_to_forget_weights,
    TFL_TensorOf<[F32, QI8]>:$recurrent_to_cell_weights,
    TFL_TensorOf<[F32, QI8]>:$recurrent_to_output_weights,

    // Cell weights
    TFL_TensorOfOrNone<[F32, QI8, QI16]>:$cell_to_input_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32, QI8, QI16]>:$cell_to_forget_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32, QI8, QI16]>:$cell_to_output_weights,

    // Bias
    TFL_TensorOfOrNone<[F32, QI32]>:$input_gate_bias,
    TFL_TensorOf<[F32, QI32]>:$forget_gate_bias,
    TFL_TensorOf<[F32, QI32]>:$cell_bias,
    TFL_TensorOf<[F32, QI32]>:$output_gate_bias,

    // Projection weight and bias
    TFL_TensorOfOrNone<[F32, QI8]>:$projection_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32, QI32]>:$projection_bias,

    // Stateful activation and cell states.
    TFL_StatefulTensor:$input_activation_state,
    TFL_StatefulTensor:$input_cell_state,

    // Layer norm coefficients
    TFL_TensorOfOrNone<[F32, QI16]>:$input_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32, QI16]>:$forget_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32, QI16]>:$cell_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32, QI16]>:$output_layer_norm_coefficients,

    // Attributes
    TFL_AFAttr:$fused_activation_function,
    ConfinedAttr<DefaultValuedOptionalAttr<F32Attr, "0.0f">, [TFL_FloatNonNegative]>:$cell_clip,
    ConfinedAttr<DefaultValuedOptionalAttr<F32Attr, "0.0f">, [TFL_FloatNonNegative]>:$proj_clip,
    // Since this op is the FULL kernel only, constrain it.
    ConfinedAttr<
      DefaultValuedOptionalAttr<TFL_LSTMKernelTypeAttr, "mlir::TFL::LSTMKernelType::FULL">,
      [TFL_LSTMKernelTypeEqualsTo<"mlir::TFL::LSTMKernelType::FULL">]>:$kernel_type,
    // Used in post-training dynamic range quantization. If the value is true,
    // input activations are asymmetrically quantized.
    OptionalAttr<BoolAttr>:$asymmetric_quantize_inputs,

    // Types of the optional intermediate tensors, which exist for fully
    // quantized LSTM op and hold the ranges of the intermediate tensors.
    // The type for intermediate tensors are be quant.calibrated when imported
    // to only store calibrated min, max values. The proper quantization spec is
    // determined while going through quantization passes.
    OptionalAttr<TypeAttr>:$input_to_input_intermediate,
    OptionalAttr<TypeAttr>:$input_to_forget_intermediate,
    OptionalAttr<TypeAttr>:$input_to_cell_intermediate,
    OptionalAttr<TypeAttr>:$input_to_output_intermediate,
    OptionalAttr<TypeAttr>:$effective_hidden_scale_intermediate
  );

  let results = (outs AnyTensor:$output);

  // TODO(fengliuai): customize printer and parser to not display
  // empty region.
  let regions = (region AnyRegion:$internal);

  let hasOptions = 1;

  let hasCanonicalizer = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    std::vector<int> GetStatefulOperands() { return {18, 19}; }
    // DynamicRangeQuantizedOpInterface:
    bool RequireAsymmetricQuantizeInputsAttr() { return true; }
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() {
      return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16};
    }
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
           LstmCifgInputConstraint,
           LstmResultConstraint,
           TFL_OperandHasRankAtLeast<0, 2>,       // input
           TFL_OperandIsNoneOrHasRank<1, 2>,      // input_to_input_weights
           TFL_OperandHasRank<2, 2>,              // input_to_forget_weights
           TFL_OperandHasRank<3, 2>,              // input_to_cell_weights
           TFL_OperandHasRank<4, 2>,              // input_to_output_weights
           TFL_OperandIsNoneOrHasRankAtMost<5, 2>,// recurrent_to_input_weights
           TFL_OperandHasRankAtMost<6, 2>,        // recurrent_to_forget_weights
           TFL_OperandHasRankAtMost<7, 2>,        // recurrent_to_cell_weights
           TFL_OperandHasRankAtMost<8, 2>,        // recurrent_to_output_weights
           TFL_OperandIsNoneOrHasRank<9, 1>,      // cell_to_input_weights
           TFL_OperandIsNoneOrHasRank<10, 1>,     // cell_to_forget_weights
           TFL_OperandIsNoneOrHasRank<11, 1>,     // cell_to_output_weights
           TFL_OperandIsNoneOrHasRank<12, 1>,     // input_gate_bias
           TFL_OperandHasRank<13, 1>,             // forget_gate_bias
           TFL_OperandHasRank<14, 1>,             // cell_gate_bias
           TFL_OperandHasRank<15, 1>,             // output_gate_bias
           TFL_OperandIsNoneOrHasRank<16, 2>,     // projection_weights
           TFL_OperandIsNoneOrHasRank<17, 1>,     // projection_bias
           TFL_StatefulOp,
           DeclareOpInterfaceMethods<InferTypeOpInterface>,
           QuantizableResult,
           DynamicRangeQuantizedOpInterface
          ]> {
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
    ins TFL_FpTensor:$input,

    // Weights
    TFL_TensorOfOrNone<[F32, QI8]>:$input_to_input_weights,
    TFL_TensorOf<[F32, QI8]>:$input_to_forget_weights,
    TFL_TensorOf<[F32, QI8]>:$input_to_cell_weights,
    TFL_TensorOf<[F32, QI8]>:$input_to_output_weights,

    // Recurrent weights
    TFL_TensorOfOrNone<[F32, QI8]>:$recurrent_to_input_weights,
    TFL_TensorOf<[F32, QI8]>:$recurrent_to_forget_weights,
    TFL_TensorOf<[F32, QI8]>:$recurrent_to_cell_weights,
    TFL_TensorOf<[F32, QI8]>:$recurrent_to_output_weights,

    // Cell weights
    TFL_TensorOfOrNone<[F32, QI8]>:$cell_to_input_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32, QI8]>:$cell_to_forget_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32, QI8]>:$cell_to_output_weights,

    // Bias
    TFL_TensorOfOrNone<[F32]>:$input_gate_bias,
    TFL_FpTensor:$forget_gate_bias,
    TFL_FpTensor:$cell_bias,
    TFL_FpTensor:$output_gate_bias,

    // Projection weight and bias
    TFL_TensorOfOrNone<[F32, QI8]>:$projection_weights,
    // Optional input
    TFL_TensorOfOrNone<[F32]>:$projection_bias,

    // Stateful activation and cell states.
    TFL_StatefulTensor:$input_activation_state,
    TFL_StatefulTensor:$input_cell_state,

    // Layer norm coefficients
    TFL_TensorOfOrNone<[F32, QI8]>:$input_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32, QI8]>:$forget_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32, QI8]>:$cell_layer_norm_coefficients,
    TFL_TensorOfOrNone<[F32, QI8]>:$output_layer_norm_coefficients,

    // Attributes
    TFL_AFAttr:$fused_activation_function,
    ConfinedAttr<DefaultValuedOptionalAttr<F32Attr, "0.0f">, [TFL_FloatNonNegative]>:$cell_clip,
    ConfinedAttr<DefaultValuedOptionalAttr<F32Attr, "0.0f">, [TFL_FloatNonNegative]>:$proj_clip,
    BoolAttr:$time_major,
    // Used in post-training dynamic range quantization. If the value is true,
    // input activations are asymmetrically quantized.
    OptionalAttr<BoolAttr>:$asymmetric_quantize_inputs,
    // IndyLSTM optimizations (i.e. optimizations enabled because of diagonal
    // recurrent weight matrices that are provided as vectors)
    OptionalAttr<BoolAttr>:$diagonal_recurrent_tensors,

    // Types of the optional intermediate tensors, which exist for fully
    // quantized op and hold the ranges of the intermediate tensors.
    // The type for intermediate tensors are be quant.calibrated when imported
    // to only store calibrated min, max values. The proper quantization spec is
    // determined while going through quantization passes.
    OptionalAttr<TypeAttr>:$input_to_input_intermediate,
    OptionalAttr<TypeAttr>:$input_to_forget_intermediate,
    OptionalAttr<TypeAttr>:$input_to_cell_intermediate,
    OptionalAttr<TypeAttr>:$input_to_output_intermediate,
    OptionalAttr<TypeAttr>:$effective_hidden_scale_intermediate
  );

  let results = (outs TFL_TensorOf<[F32, QI8]>:$output);

  let hasOptions = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    std::vector<int> GetStatefulOperands() { return {18, 19}; }

    // Compatiable return types check
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);

    // DynamicRangeQuantizedOpInterface:
    bool RequireAsymmetricQuantizeInputsAttr() { return true; }
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() {
      return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16};
    }
  }];
}

def BidiLstmMandatoryInputsConstraint : PredOpTrait<
  "mandatory operands element types should match",
  // TODO(ashwinm): Replace the indices with input tensor names when that
  // support is available.
  Or<[
    TCopVTEtAreSameAt<[0, 2, 3, 4, 6, 7, 8, 13, 14, 15, 19, 20, 21, 23, 24, 25,
                       30, 31, 32, 35, 36, 37, 38]>,
    Neg<TypeIsPred<"input", F32>>]>>;

// TODO(b/172517537): support quantized types
def BidiLstmOptionalPeepholeWeightConstraint : PredOpTrait<
  "the optional peephole weights should all be specified or none",
  TCopVTEtAreSameAt<[9, 10, 11, 26, 27, 28]>>;

def BidiLstmProjectionWeightBiasConstraint : PredOpTrait<
  "either projection weight must be specified or both projection weight and "
  "projection bias must not be specified",
   Or<[
      And<[TypeIsPred<"fw_projection_weights", NoneType>,
           TypeIsPred<"fw_projection_bias", NoneType>,
           TypeIsPred<"bw_projection_weights", NoneType>,
           TypeIsPred<"bw_projection_bias", NoneType>]>,
      And<[
        Neg<TypeIsPred<"fw_projection_weights", NoneType>>,
        Neg<TypeIsPred<"bw_projection_weights", NoneType>>,
     ]>
   ]>>;

// BidirectionalSequenceLstm op.
// TODO(ashwinm): Add constraint to validate the combination of operands
// that are valid for hybrid vs fully quantized vs float only semantics
def TFL_BidirectionalSequenceLSTMOp :
  TFL_Op<"bidirectional_sequence_lstm",
          [BidiLstmMandatoryInputsConstraint,
           BidiLstmOptionalPeepholeWeightConstraint,
           BidiLstmProjectionWeightBiasConstraint,
           LstmResultConstraint,
           TFL_OperandHasRank<0, 3>,   // input
           TFL_OperandHasRank<1, 2>,   // fw_input_to_input_weights
           TFL_OperandHasRank<2, 2>,   // fw_input_to_forget_weights
           TFL_OperandHasRank<3, 2>,   // fw_input_to_cell_weights
           TFL_OperandHasRank<4, 2>,   // fw_input_to_output_weights
           TFL_OperandHasRank<5, 2>,   // fw_recurrent_to_input_weights
           TFL_OperandHasRank<6, 2>,   // fw_recurrent_to_forget_weights
           TFL_OperandHasRank<7, 2>,   // fw_recurrent_to_cell_weights
           TFL_OperandHasRank<8, 2>,   // fw_recurrent_to_output_weights
           TFL_OperandHasRank<9, 1>,   // fw_cell_to_input_weights
           TFL_OperandHasRank<10, 1>,  // fw_cell_to_forget_weights
           TFL_OperandHasRank<11, 1>,  // fw_cell_to_output_weights
           TFL_OperandHasRank<12, 1>,  // fw_input_gate_bias
           TFL_OperandHasRank<13, 1>,  // fw_forget_gate_bias
           TFL_OperandHasRank<14, 1>,  // fw_cell_bias
           TFL_OperandHasRank<15, 1>,  // fw_output_gate_bias
           TFL_OperandHasRank<16, 2>,  // fw_projection_weights
           TFL_OperandHasRank<17, 1>,  // fw_projection_bias
           TFL_OperandHasRank<18, 2>,  // bw_input_to_input_weights
           TFL_OperandHasRank<19, 2>,  // bw_input_to_forget_weights
           TFL_OperandHasRank<20, 2>,  // bw_input_to_cell_weights
           TFL_OperandHasRank<21, 2>,  // bw_input_to_output_weights
           TFL_OperandHasRank<22, 2>,  // bw_recurrent_to_input_weights
           TFL_OperandHasRank<23, 2>,  // bw_recurrent_to_forget_weights
           TFL_OperandHasRank<24, 2>,  // bw_recurrent_to_cell_weights
           TFL_OperandHasRank<25, 2>,  // bw_recurrent_to_output_weights
           TFL_OperandHasRank<26, 1>,  // bw_cell_to_input_weights
           TFL_OperandHasRank<27, 1>,  // bw_cell_to_forget_weights
           TFL_OperandHasRank<28, 1>,  // bw_cell_to_output_weights
           TFL_OperandHasRank<29, 1>,  // bw_input_gate_bias
           TFL_OperandHasRank<30, 1>,  // bw_forget_gate_bias
           TFL_OperandHasRank<31, 1>,  // bw_cell_bias
           TFL_OperandHasRank<32, 1>,  // bw_output_gate_bias
           TFL_OperandHasRank<33, 2>,  // bw_projection_weights
           TFL_OperandHasRank<34, 1>,  // bw_projection_bias
           TFL_StatefulOp,
           QuantizableResult,
           DynamicRangeQuantizedOpInterface]> {
  let summary = "Bidirectional sequence lstm operator";

  let description = [{
    Bidirectional lstm is essentially two lstms, one running forward & the
    other running backward. And the output is the concatenation of the two
    lstms.
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, I8]>:$input,

    // Forward LSTM Weights
    TFL_TensorOfOrNone<[F32, I8]>:$fw_input_to_input_weights,
    TFL_TensorOf<[F32, I8]>:$fw_input_to_forget_weights,
    TFL_TensorOf<[F32, I8]>:$fw_input_to_cell_weights,
    TFL_TensorOf<[F32, I8]>:$fw_input_to_output_weights,

    // Forward Recurrent weights
    TFL_TensorOfOrNone<[F32, I8]>:$fw_recurrent_to_input_weights,
    TFL_TensorOf<[F32, I8]>:$fw_recurrent_to_forget_weights,
    TFL_TensorOf<[F32, I8]>:$fw_recurrent_to_cell_weights,
    TFL_TensorOf<[F32, I8]>:$fw_recurrent_to_output_weights,

    // Forward Cell weights
    TFL_TensorOfOrNone<[F32, I8]>:$fw_cell_to_input_weights,
    // Optional Forward cell weights
    TFL_TensorOfOrNone<[F32, I8]>:$fw_cell_to_forget_weights,
    // Optional Forward cell weights
    TFL_TensorOfOrNone<[F32, I8]>:$fw_cell_to_output_weights,

    // Forward Bias
    TFL_TensorOfOrNone<[F32]>:$fw_input_gate_bias,
    TFL_TensorOf<[F32]>:$fw_forget_gate_bias,
    TFL_TensorOf<[F32]>:$fw_cell_bias,
    TFL_TensorOf<[F32]>:$fw_output_gate_bias,

    // Forward Projection weight and bias
    TFL_TensorOfOrNone<[F32, I8]>:$fw_projection_weights,
    // Forward Optional input
    TFL_TensorOfOrNone<[F32]>:$fw_projection_bias,

    // Backward LSTM Weights
    TFL_TensorOfOrNone<[F32, I8]>:$bw_input_to_input_weights,
    TFL_TensorOf<[F32, I8]>:$bw_input_to_forget_weights,
    TFL_TensorOf<[F32, I8]>:$bw_input_to_cell_weights,
    TFL_TensorOf<[F32, I8]>:$bw_input_to_output_weights,

    // Backward Recurrent weights
    TFL_TensorOfOrNone<[F32, I8]>:$bw_recurrent_to_input_weights,
    TFL_TensorOf<[F32, I8]>:$bw_recurrent_to_forget_weights,
    TFL_TensorOf<[F32, I8]>:$bw_recurrent_to_cell_weights,
    TFL_TensorOf<[F32, I8]>:$bw_recurrent_to_output_weights,

    // Backward Cell weights
    TFL_TensorOfOrNone<[F32, I8]>:$bw_cell_to_input_weights,
    // Optional Forward cell weights
    TFL_TensorOfOrNone<[F32, I8]>:$bw_cell_to_forget_weights,
    // Optional Forward cell weights
    TFL_TensorOfOrNone<[F32, I8]>:$bw_cell_to_output_weights,

    // Backward Bias
    TFL_TensorOfOrNone<[F32]>:$bw_input_gate_bias,
    TFL_TensorOf<[F32]>:$bw_forget_gate_bias,
    TFL_TensorOf<[F32]>:$bw_cell_bias,
    TFL_TensorOf<[F32]>:$bw_output_gate_bias,

    // Backward Projection weight and bias
    TFL_TensorOfOrNone<[F32, I8]>:$bw_projection_weights,
    // Backward Optional input
    TFL_TensorOfOrNone<[F32]>:$bw_projection_bias,

    // Stateful activation and cell states.
    TFL_StatefulTensor:$fw_input_activation_state,
    TFL_StatefulTensor:$fw_input_cell_state,
    TFL_StatefulTensor:$bw_input_activation_state,
    TFL_StatefulTensor:$bw_input_cell_state,

    // Auxiliary input & weights.
    TFL_TensorOfOrNone<[F32, I8]>:$aux_input,
    // Auxiliary fw weights.
    TFL_TensorOfOrNone<[F32, I8]>:$fw_aux_input_to_input_weights,
    TFL_TensorOfOrNone<[F32, I8]>:$fw_aux_input_to_forget_weights,
    TFL_TensorOfOrNone<[F32, I8]>:$fw_aux_input_to_cell_weights,
    TFL_TensorOfOrNone<[F32, I8]>:$fw_aux_input_to_output_weights,
    // Auxiliary bw weights.
    TFL_TensorOfOrNone<[F32, I8]>:$bw_aux_input_to_input_weights,
    TFL_TensorOfOrNone<[F32, I8]>:$bw_aux_input_to_forget_weights,
    TFL_TensorOfOrNone<[F32, I8]>:$bw_aux_input_to_cell_weights,
    TFL_TensorOfOrNone<[F32, I8]>:$bw_aux_input_to_output_weights,

    // Attributes
    TFL_AFAttr:$fused_activation_function,
    ConfinedAttr<DefaultValuedOptionalAttr<F32Attr, "0.0f">, [TFL_FloatNonNegative]>:$cell_clip,
    ConfinedAttr<DefaultValuedOptionalAttr<F32Attr, "0.0f">, [TFL_FloatNonNegative]>:$proj_clip,
    BoolAttr:$merge_outputs,
    BoolAttr:$time_major,
    // Used in post-training dynamic range quantization. If the value is true,
    // input activations are asymmetrically quantized.
    OptionalAttr<BoolAttr>:$asymmetric_quantize_inputs
  );

  let results = (outs
    AnyTensor:$fw_output,
    AnyTensor:$bw_output
  );

  let hasOptions = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    std::vector<int> GetStatefulOperands() { return {35, 36, 37, 38}; }

    // DynamicRangeQuantizedOpInterface:
    bool RequireAsymmetricQuantizeInputsAttr() { return true; }
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() {
      return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 18, 19, 20, 21,
              22, 23, 24, 25, 26, 27, 28, 33, 40, 41, 42, 43, 44, 45, 46, 47};
    }
  }];
}

// UnidirectionalSequenceRNN op.
def TFL_UnidirectionalSequenceRNNOp : TFL_Op<"unidirectional_sequence_rnn", [
    TFL_OperandHasRank<4, 2>,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    PredOpTrait<"input and constant value operands must have same element type",
      TFL_TCopVTEtAreSameAt<1, 2>>,
    TFL_StatefulOp,
    QuantizableResult,
    DynamicRangeQuantizedOpInterface]> {
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
    ins TFL_FpTensor:$input,

    // Weights
    TFL_TensorOf<[F32, QI8]>:$input_to_input_weights,

    // Recurrent weights
    TFL_TensorOf<[F32, QI8]>:$recurrent_to_input_weights,

    // Bias
    TFL_FpTensor:$input_gate_bias,

    // Hidden state.
    TFL_StatefulTensor:$hidden_state,

    // Attributes
    BoolAttr:$time_major,
    TFL_AFAttr:$fused_activation_function,
    // Used in post-training dynamic range quantization. If the value is true,
    // input activations are asymmetrically quantized.
    OptionalAttr<BoolAttr>:$asymmetric_quantize_inputs
  );

  let results = (outs TFL_FpTensor:$output);

  let hasOptions = 1;

  let customOption = "SequenceRNNOptions";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    std::vector<int> GetStatefulOperands() { return {4}; }

    // DynamicRangeQuantizedOpInterface:
    bool RequireAsymmetricQuantizeInputsAttr() { return true; }
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() {
      return {1, 2};
    }
  }];
}

def TFL_WhereOp : TFL_Op<"where", [Pure]> {
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
    TFL_TensorOf<[I1, F32, TFL_Int32Or64, I8, UI8, UI32]>:$condition
  );

  let results = (outs
    TFL_I64Tensor:$index
  );
}

def TFL_NumericVerifyOp : Op<TFL_Dialect, "NumericVerify", [
    QuantizableResult,
    SameOperandsShape,
    DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {

  let summary = "Verifies the numericals of the two operands";

  let description = [{
    The NumericVerify op is a debugging op to verify the numericals of the two
    activations. It is a custom op in TFLite.
    If log_if_failed is true, the NumericVerify op calculates statistics on
    differences between float and quantized activations, output
    logs, set differences to the output tensors, and throws an error if errors
    above tolerance exist. If log_if_failed = false, then it doesn't care about
    errors.
  }];

  let arguments = (ins
    TFL_TensorOf<[QI8, QUI8, QI16, F16, TFL_Quint8]>:$input,
    TFL_TensorOf<[F32]>:$ref,

    // Attributes
    DefaultValuedOptionalAttr<F32Attr, "0.1">:$tolerance,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$log_if_failed
  );

  let results = (outs TFL_FpTensor:$output);
}

// SVDF op.
def TFL_SVDFOp :
  TFL_Op<"svdf", [
    PredOpTrait<"the input and result tensor elemental types must be same",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_StatefulOp,
    AccumulatorUniformScale<3, 2, 4>,
    QuantizableResult,
    DynamicRangeQuantizedOpInterface]> {

  let summary = "Single value decomposition filter operator";

  let description = [{
    The SVDF op is a decomposition of a densely connected op into low rank
    filters.
    For details: https://research.google.com/pubs/pub43813.html
                 https://arxiv.org/abs/1812.02802
  }];

  let arguments = (
    ins TFL_TensorOf<[F32, QI8]>:$input,

    // Feature Weights.
    TFL_TensorOf<[F32, QI8, QUI8]>:$feature_weights,

    // Time weights
    TFL_TensorOf<[F32, QI16]>:$time_weights,

    // Bias
    TFL_TensorOfOrNone<[F32, QI32]>:$input_gate_bias,

    // Activation state.
    TFL_StatefulTensor:$activation_state,

    // Attributes
    ConfinedAttr<I32Attr, [IntPositive]>:$rank,
    TFL_AFAttr:$fused_activation_function,
    // Used in post-training dynamic range quantization. If the value is true,
    // input activations are asymmetrically quantized.
    OptionalAttr<BoolAttr>:$asymmetric_quantize_inputs
  );

  let results = (outs TFL_TensorOf<[F32, QI8]>:$output);

  let hasOptions = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // StatefulOpInterface:
    std::vector<int> GetStatefulOperands() { return {4}; }

    // DynamicRangeQuantizedOpInterface:
    bool RequireAsymmetricQuantizeInputsAttr() { return true; }
    bool GetDynamicRangeQuantKernelSupport() { return true; }
    std::vector<int> GetQuantizableOperandIndices() {
      return {1, 2};
    }
  }];
}

def TFL_SegmentSumOp: TFL_Op<"segment_sum", [
    Pure,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "SegmentSum operator";

  let description = [{
    Computes the sum along segments of a tensor.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32]>:$input,
    TFL_I32Tensor:$segment_ids
  );
  let results = (outs TFL_TensorOf<[F32, I32]>:$output);
}

def TFL_UnsortedSegmentProdOp: TFL_Op<"unsorted_segment_prod", [
    Pure,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {
  let summary = "UnsortedSegmentProd operator";

  let description = [{
    Computes the product along segments of a tensor.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32]>:$input,
    TFL_I32Tensor:$segment_ids,
    TFL_I32Tensor:$num_segments
  );
  let results = (outs TFL_TensorOf<[F32, I32]>:$output);
}

def TFL_UnsortedSegmentMaxOp: TFL_Op<"unsorted_segment_max", [
    Pure, PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {

  let summary = "UnsortedSegmentMax operator";

  let description = [{
    Computes the maximum value along segments of a tensor such that
    output[i] = max(data[j....]) where segment_ids[j...] = i
    if the maximum is empty for a given segment ID i,
    it outputs the smallest possible value for the specific numeric type,
    output[i] = numeric_limits::lowest().
    Note the values of segment_ids are always validated to be less than
    num_segments and an error is thrown for out-of-bound indices.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32]>:$input,
    TFL_I32Tensor:$segment_ids,
    TFL_I32Tensor:$num_segments
  );

  let results = (outs TFL_TensorOf<[F32, I32]>:$output);
}

def TFL_UnsortedSegmentMinOp: TFL_Op<"unsorted_segment_min", [
    Pure, PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {

  let summary = "UnsortedSegmentMin operator";

  let description = [{
    Computes the minimum value along segments of a tensor such that
    output[i] = min(data[j....]) where segment_ids[j...] = i
    if the minimum is empty for a given segment ID i,
    it outputs the largest possible value for the specific numeric type,
    output[i] = numeric_limits::max().
    Note the values of segment_ids are always validated to be less than
    num_segments and an error is thrown for out-of-bound indices.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32]>:$input,
    TFL_I32Tensor:$segment_ids,
    TFL_I32Tensor:$num_segments
  );

  let results = (outs TFL_TensorOf<[F32, I32]>:$output);
}

def TFL_UnsortedSegmentSumOp: TFL_Op<"unsorted_segment_sum", [
    Pure, PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>]> {

  let summary = "UnsortedSegmentSum operator";

  let description = [{
    From a tensor segmentation, computes the `output` resulting from
    summing together elements mapped to the same segment_id. I.e. `output[i]` is
    equal to the tensor sum of all elements from the input tensor mapped to
    segment_id `i`. If no tensors are mapped to a particular included
    segment_id, the output at that indice will be a zero tensor with the
    appropriate shape. Note the values of segment_ids are always validated to be
    less than num_segments and an error is thrown for out-of-bound indices
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32]>:$input,
    TFL_I32Tensor:$segment_ids,
    TFL_I32Tensor:$num_segments
  );

  let results = (outs TFL_TensorOf<[F32, I32]>:$output);
}

def TFL_Atan2Op: TFL_Op<"atan2", [
  Pure,
  SameOperandsAndResultShape,
  SameOperandsAndResultElementType]> {

  let summary = "Atan2 operation";
  let description = [{
    The "atan2" operation computes the arctangent of y/x element-wise,
    respecting signs of the arguments.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, F64]>:$y,
    TFL_TensorOf<[F32, F64]>:$x
  );

  let results = (outs
    TFL_TensorOf<[F32, F64]>:$output
  );
}

def TFL_SignOp: TFL_Op<"sign", [
  Pure,
  SameOperandsAndResultShape,
  SameOperandsAndResultElementType
  ]> {

  let summary = "Sign operation";
  let description = [{
    Returns NaN if x is NaN, 0 if x is 0, -1 if x < 0 and 1 if x > 0.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, F64, I32]>:$x
  );

  let results = (outs
    TFL_TensorOf<[F32, F64, I32]>:$output
  );
}

def TFL_YieldOp : Op<TFL_Dialect, "yield",
  [Pure,
   Terminator,
   QuantizableResult,
   DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {
  let summary = "Yield operation";
  let description = [{
    The "yield" operation represents a return operation within the conditional
    and body of structured control flow (e.g., while), and a terminator for ControlNodeOp.
    The operation takes a variable number of operands and produces no results.
    The operand number and types must match the signature of the region that contains the operation.
  }];
  let arguments = (ins Variadic<AnyType>);

  // Default builder needed for ensureTerminator
  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, {});
    }]>
  ];
}

def TFL_IfOp : Op<TFL_Dialect, "if", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveMemoryEffects,
    NoRegionArguments,
    DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {
  let summary = [{if-then-else operation}];

  let description = [{
    The `tfl.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    tfl.if %b  {
      ...
    } else {
      ...
    }
    ```

    `tfl.if` may also return results that are defined in its regions. The
    values defined are determined by which execution path is taken.

    Example:

    ```mlir
    %x, %y = tfl.if %b -> (tensor<f32>, tensor<f32>) {
      %x_true = ...
      %y_true = ...
      tfl.yield %x_true, %y_true : tensor<f32>, tensor<f32>
    } else {
      %x_false = ...
      %y_false = ...
      tfl.yield %x_false, %y_false : tensor<f32>, tensor<f32>
    }
    ```

    `tfl.if` regions are always terminated with "tfl.yield". If "tfl.if"
    defines no values, the "tfl.yield" can be left out, and will be inserted
    implicitly. Otherwise, it must be explicit.
    Also, if "tfl.if" defines one or more values, the 'else' block cannot be
    omitted.

    Example:

    ```mlir
    tfl.if %b  {
      ...
    }
    ```
  }];

  let arguments = (ins TFL_BoolTensor:$cond);
  let results = (outs Variadic<AnyTensor>:$results);
  let regions = (region SizedRegion<1>:$then_region, AnyRegion:$else_region);

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                               : OpBuilder::atBlockEnd(body, listener);
    }
    OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(1);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                               : OpBuilder::atBlockEnd(body, listener);
    }
  }];

  // Canonicalizer wasn't defined for this one. In practise, we legalize the
  // tf.IfOp to scf.If op first and then legalize it to tfl.if to reduce
  // code redundancy.
}

def TFL_WhileOp : Op<TFL_Dialect, "while", [
    DeclareOpInterfaceMethods<LoopLikeOpInterface, ["isDefinedOutsideOfLoop"]>,
    SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {
  let summary = [{While loop}];

  let description = [{
    output = input; while (cond(output)) { output = body(output) }

    While loop where all values are passes through arguments with implicit
    capture.

    input: A list of input tensors whose types are T.
    output: A list of output tensors whose types are T.
    cond: A region that takes 'input' and returns a boolean scalar tensor.
    body: A region that takes a list of tensors and returns another
          list of tensors. Both lists have the same types.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$input,

    // Used to map StatelessWhile and While op defined in TensorFlow to a common
    // op.
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_stateless
  );
  let results = (outs Variadic<AnyTensor>:$output);

  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TFL_CallOnceOp : TFL_Op<"call_once", []> {
  let summary = "Invokes an initialization function";

  let description = [{
This operation invokes the given initialization function for the session
initializer in tf saved model dialect.
  }];

  let arguments = (ins
    StrAttr:$session_init_function
  );

  let results = (outs);
}

def TFL_CustomOp : Op<TFL_Dialect, "custom", [
  DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {
  let summary = "Custom op";

  let description = [{
    A generic op for any TFLite custom operation.

    input: A list of inputs in the original op.
    custom_code: A string used to identify which exactly this op is, which
                 corresponds to operator_codes.custom_code in the flatbuffer.
    custom_option: a holder to save the op attributes in bytes fashion.
    output: A list of outputs in the original op.
  }];

  let arguments = (ins
    Variadic<TFL_TensorOfOrNone<[AnyType]>>:$input,
    StrAttr:$custom_code,
    TFL_ConstBytesAttr:$custom_option
  );
  let results = (outs Variadic<AnyTensor>:$output);

  let hasVerifier = 1;
}

def TFL_CustomTfOp : Op<TFL_Dialect, "custom_tf", [
  RecursiveMemoryEffects,
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"YieldOp">,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  DeclareOpInterfaceMethods<TFL_RuntimeVerification>]> {
  let summary = "Wrapper Op for TF custom ops.";

  let description = [{
    A wrapper op around any Custom TF op. These includes ops defined using
    custom_opdefs or linked which are not defined in TF dialect.
    This Op just wraps the custom op inside a region.
    Note #1, this Op will not include TF Lite custom ops defined using CustomOp.
    Note #2, this op is just internal representation inside the converter and
    are not exposed/exported when the model is exported to Flatbuffer.
  }];

  let arguments = (ins
    Variadic<TFL_TensorOfOrNone<[AnyType]>>:$input
  );
  let results = (outs Variadic<AnyTensor>:$output);

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

def TFL_BroadcastToOp : TFL_Op<"broadcast_to", [
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandHasRankAtMost<0, 8>,
    TFL_OperandHasRank<1, 1>,
    PredOpTrait<"output dimension count must be at most 8",
      Or<[TFL_OperandIsUnrankedPred<1>,
          TFL_OperandDimIsAtMost<1, 0, 8>]>>,
    QuantizableResult,
    Pure]> {
  let summary = "Broadcast an array for a compatible shape.";

  let description = [{
Broadcasting is the process of making arrays to have compatible shapes
for arithmetic operations. Two shapes are compatible if for each
dimension pair they are either equal or one of them is one. When trying
to broadcast a Tensor to a shape, it starts with the trailing dimensions,
and works its way forward.

For example,

>>> x = tf.constant([1, 2, 3])
>>> y = tf.broadcast_to(x, [3, 3])
>>> print(y)
tf.Tensor(
    [[1 2 3]
     [1 2 3]
     [1 2 3]], shape=(3, 3), dtype=int32)

In the above example, the input Tensor with the shape of `[1, 3]`
is broadcasted to output Tensor with shape of `[3, 3]`.

When doing broadcasted operations such as multiplying a tensor
by a scalar, broadcasting (usually) confers some time or space
benefit, as the broadcasted tensor is never materialized.

However, `broadcast_to` does not carry with it any such benefits.
The newly-created tensor takes the full memory of the broadcasted
shape. (In a graph context, `broadcast_to` might be fused to
subsequent operation and then be optimized away, however.)
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, I32, I1, I8, QI8, UI8, QUI8, I16, QI16, I64, Complex<F<32>>]>:$input,
    TFL_I32OrI64Tensor:$shape
  );

  let results = (outs
    TFL_TensorOf<[F32, I32, I1, I8, QI8, UI8, QUI8, I16, QI16, I64, Complex<F<32>>]>:$output
  );

  let hasCanonicalizer = 1;
}

def TFL_RFFT2dOp : TFL_Op<"rfft2d", [Pure]> {
  let summary = "2D real-valued fast Fourier transform.";

  let description = [{
Computes the 2-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 2 dimensions of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

Along each axis `RFFT2D` is computed on, if `fft_length` is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.
  }];

  let arguments = (ins
    TFL_FpTensor:$input,
    TFL_I32Tensor:$fft_length
  );

  let results = (outs
    TFL_Complex64Tensor:$output
  );
}

def TFL_VarHandleOp : TFL_Op<"var_handle", []> {
  let summary = "Returns a handle to a variable resource from its name.";

  let description = [{
    Returns a handle for a variable resource from its name.
    container: the container this variable is placed in.
    shared_name: the name by which this variable is referred to.
  }];

  let arguments = (ins
    DefaultValuedStrAttr<StrAttr, "">:$container,
    DefaultValuedStrAttr<StrAttr, "">:$shared_name
  );

  let results = (outs TFL_ResourceTensor:$resource_handle);

  let hasOptions = 1;
}

def TFL_AssignVariableOp : TFL_Op<"assign_variable", []> {
  let summary = "Assigns a new value to a variable.";

  let description = [{
Any ReadVariableOp with a control dependency on this op is guaranteed to return
this value or a subsequent newer value of the variable.
  }];

  let arguments = (ins
    TFL_ResourceTensor:$resource_id,
    TFL_TensorOf<[F32, F64, I1, UI8, I8, QI8, QUI8, I32, I64, QI16, Complex<F<32>>, Complex<F<64>>]>:$value
  );

  let results = (outs);
}

def TFL_ReadVariableOp : TFL_Op<"read_variable", []> {
  let summary = "Reads variable value.";

  let description = [{
Read variable data identified by 'resource_id'.
  }];

  let arguments = (ins
    TFL_ResourceTensor:$resource_id
  );

  let results = (outs TFL_TensorOf<[F32, F64, I1, UI8, I8, QI8, QUI8, I32, I64, QI16, Complex<F<32>>, Complex<F<64>>]>:$result);
}

def TFL_Conv3DOp : TFL_Op<"conv_3d", [
    Pure,
    AccumulatorUniformScale<2, 0, 1>,
    TFL_OperandHasRank<0, 5>,
    TFL_OperandHasRank<1, 5>,
    // Channel dimension in input and filter should match.
    TFL_OperandsHaveSameDimsTrait<0, 1, 4, 3>,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    PredOpTrait<"bias and output must have same element type",
      Or<[
        TFL_OperandIsNoneType<2>,
        TFL_TCresVTEtIsSameAsOp<0, 2>]>>,
    PredOpTrait<"bias must has num of elements equals to 4th dim of filter",
      Or<[
        TFL_OperandIsNoneType<2>,
        TFL_NumElementsEqualsDim<2, 1, 4>]>>]> {
  let summary = "Convolution 3D operator";

  let description = [{
    Performs convolution operation on 3D inputs.
    Inputs:
      `inputs[0]`: required: the input activation tensor
      `inputs[1]`: required: the filter weight tensor
      `inputs[2]`: optional: the bias tensor
  }];

  let arguments = (ins
    TFL_TensorOf<[F32]>:$input,
    TFL_TensorOf<[F32]>:$filter,
    TFL_TensorOfOrNone<[F32]>:$bias,
    I32Attr:$dilation_d_factor,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    TFL_AFAttr:$fused_activation_function,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_d,
    I32Attr:$stride_h,
    I32Attr:$stride_w
  );

  let results = (outs TFL_TensorOf<[F32]>:$output);

  let hasOptions = 1;

  let customOption = "Conv3DOptions";
}

def TFL_Conv3DTransposeOp : TFL_Op<"conv_3d_transpose", [
    Pure,
    AccumulatorUniformScale<2, 0, 1>,
    TFL_OperandHasRank<0, 1>,
    TFL_OperandHasRank<1, 5>,
    TFL_OperandHasRank<2, 5>,
    TFL_NumElementsTrait<0, 5>,
    // Channel dimension in input and filter should match.
    TFL_OperandsHaveSameDimsTrait<2, 1, 4, 4>,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 2>>,
    PredOpTrait<"bias and output must have same element type",
      Or<[
        TFL_OperandIsNoneType<3>,
        TFL_TCresVTEtIsSameAsOp<0, 3>]>>,
    PredOpTrait<"bias must has num of elements equals to 4th dim of filter",
      Or<[
        TFL_OperandIsNoneType<3>,
        TFL_NumElementsEqualsDim<3, 1, 4>]>>]> {
  let summary = "Transposed Convolution 3D operator";

  let description = [{
    Performs transposed convolution operation on 3D inputs.
    Inputs:
      `inputs[0]`: required: the shape of output tensor
      `inputs[1]`: required: the filter weight tensor
      `inputs[2]`: required: the input activation tensor
      `inputs[3]`: optional: the bias tensor
  }];

  let arguments = (ins
    TFL_I32Tensor:$output_shape,
    TFL_TensorOf<[F32]>:$filter,
    TFL_TensorOf<[F32]>:$input,
    TFL_TensorOfOrNone<[F32]>:$bias,
    I32Attr:$dilation_d_factor,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    TFL_AFAttr:$fused_activation_function,
    TFL_PaddingAttr:$padding,
    I32Attr:$stride_d,
    I32Attr:$stride_h,
    I32Attr:$stride_w
  );

  let results = (outs TFL_TensorOf<[F32]>:$output);

  let hasOptions = 1;

  let customOption = "Conv3DOptions";
}

def TFL_ComplexAbsOp : TFL_Op<"complex_abs", [
  Pure,
  SameOperandsAndResultShape]> {
  let summary = "Computes the complex absolute value of a tensor.";

  let description = [{
Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` or `double` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
value is computed as \\( \sqrt{a^2 + b^2}\\).
  }];

  let arguments = (ins
    TFL_TensorOf<[Complex<F<32>>, Complex<F<64>>]>:$input
  );

  let results = (outs
    TFL_TensorOf<[F32, F64]>:$output
  );
}

def TFL_RealOp : TFL_Op<"real", [
  Pure,
  SameOperandsAndResultShape]> {
  let summary = "Returns the real part of a complex number.";

  let description = [{
Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the real part of each element in `input`. All elements in
`input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
 part returned by this operation and *b* is the imaginary part.
  }];

  let arguments = (ins
    TFL_TensorOf<[Complex<F<32>>, Complex<F<64>>]>:$input
  );

  let results = (outs
    TFL_TensorOf<[F32, F64]>:$output
  );
}

def TFL_ImagOp : TFL_Op<"imag", [
  Pure,
  SameOperandsAndResultShape]> {
  let summary = "Returns the imaginary part of a complex number.";

  let description = [{
Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the imaginary part of each element in `input`. All
elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
is the real part and *b* is the imaginary part returned by this operation.
  }];

  let arguments = (ins
    TFL_TensorOf<[Complex<F<32>>, Complex<F<64>>]>:$input
  );

  let results = (outs
    TFL_TensorOf<[F32, F64]>:$output
  );
}

def TFL_HashtableOp: TFL_Op<"hashtable", []> {
  let summary = "Creates a non-initialized hash table.";
  let description = [{
This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.
  }];

  let arguments = (ins
    I32Attr:$table_id,
    TypeAttr:$key_dtype,
    TypeAttr:$value_dtype
  );

  let results = (outs TFL_ResourceTensor:$out);

  let hasOptions = 1;
}

def TFL_HashtableFindOp: TFL_Op<"hashtable_find", []> {
  let summary = "Looks up keys in a table, outputs the corresponding values.";

  let description = [{
The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.
  }];

  let arguments = (ins
    TFL_ResourceTensor:$hash_table,
    TFL_TensorOf<[I32, TFL_Str, I64]>:$keys,
    TFL_TensorOf<[F32, I32, TFL_Str, I64]>:$default_value
  );

  let results = (outs TFL_TensorOf<[F32, I32, TFL_Str, I64]>:$out);
}

def TFL_HashtableImportOp: TFL_Op<"hashtable_import", []> {
  let summary = [{
Replaces the contents of the table with the specified keys and values.
  }];

  let description = [{
The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.
  }];

  let arguments = (ins
    TFL_ResourceTensor:$hash_table,
    TFL_TensorOf<[I32, TFL_Str, I64]>:$keys,
    TFL_TensorOf<[F32, I32, TFL_Str, I64]>:$values
  );

  let results = (outs);
}


def TFL_HashtableSizeOp: TFL_Op<"hashtable_size", []> {
  let summary = "Computes the number of elements in the given table.";

  let arguments = (ins
    TFL_ResourceTensor:$hash_table
  );

  let results = (outs
    TFL_I64Tensor:$out
  );
}

def TFL_BroadcastArgsOp : TFL_Op<"broadcast_args",[
    OperandsSameElementTypeConstraintBase<"BroadcastArgs op">,
    PredOpTrait<"input and output must have same element type",
      TFL_TCresVTEtIsSameAsOp<0, 0>>,
    TFL_OperandHasRank<0, 1>,
    TFL_OperandHasRank<1, 1>,
    Pure]> {
  let summary = "Return the shape of s0 op s1 with broadcast.";

  let description = [{
Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
  }];

  let arguments = (ins
    TFL_I32OrI64Tensor:$s0,
    TFL_I32OrI64Tensor:$s1
  );

  let results = (outs
    TFL_I32OrI64Tensor:$r0
  );
}

def TFL_BucketizeOp
    : TFL_Op<"bucketize", [Pure, SameOperandsAndResultShape]> {
  let summary = "Bucketizes 'input' based on 'boundaries'.";

  let description = [{
Example:

If the inputs are `boundaries = [0, 10, 100]` and
`input = [[-5, 10000][150, 10][5, 100]]`,
then the output will be `output = [[0, 3][3, 2][1, 3]]`.
  }];

  let arguments = (ins
    TFL_TensorOf<[F32, F64, I32, I64]>:$input,
    F32ArrayAttr:$boundaries
  );

  let results = (outs
    TFL_TensorOf<[I32]>:$output
  );

  let hasOptions = 1;
}

def TFL_RandomUniformOp : TFL_Op<"random_uniform", []> {
  let summary = "Outputs random values from a uniform distribution.";

  let description = [{
The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.
  }];

  let arguments = (ins
    TFL_I32Tensor:$shape,
    DefaultValuedOptionalAttr<I64Attr, "0">:$seed,
    DefaultValuedOptionalAttr<I64Attr, "0">:$seed2
  );

   let results = (outs
    TFL_TensorOf<[F32]>:$out);
  let hasOptions = 1;
  let customOption = "RandomOptions";
}

def TFL_RandomStandardNormalOp : TFL_Op<"random_standard_normal", []> {
  let summary = "Outputs random values from a normal distribution.";

  let description = [{
The generated values will have mean 0 and standard deviation 1.
  }];

  let arguments = (ins
    TFL_I32Tensor:$shape,
    DefaultValuedOptionalAttr<I64Attr, "0">:$seed,
    DefaultValuedOptionalAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    TFL_TensorOf<[F32]>:$out);
  let hasOptions = 1;
  let customOption = "RandomOptions";
}

def TFL_MultinomialOp : TFL_Op<"multinomial", []> {
  let summary = "Draws samples from a categorical distribution.";

  let description = [{
The generated values will have a categorical distribution based on the `logits`
or unnormalized log-probabilities provided for all classes.
  }];

  let arguments = (ins
    TFL_FpTensor:$logits,
    TFL_I32Tensor:$num_samples,
    DefaultValuedOptionalAttr<I64Attr, "0">:$seed,
    DefaultValuedOptionalAttr<I64Attr, "0">:$seed2
  );

  let results = (outs
    TFL_TensorOf<[I32, I64]>:$out);

  let hasOptions = 1;
  let customOption = "RandomOptions";
}

def TFL_NoValueOp : Op<TFL_Dialect, "no_value", [ConstantLike, Pure]> {
  let summary = "constant representing no value.";

  let description = [{
No value constant op.
  }];

  let arguments = (ins UnitAttr:$value);

  let results = (outs NoneType:$none_val);

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];
}

def TFL_ControlNodeOp : Op<TFL_Dialect, "control_node",
    [HasParent<"mlir::func::FuncOp">, RecursiveMemoryEffects, SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = [{
    The `TFL.control_node` operation wraps single-block operations in order to attach control edges.
  }];

  let description = [{
    This is used to wrap regions and attach control dependencies to them. Typically,
    this will happen in one of the last steps before emitting the flatbuffer model
    in order to enable optimizations that rely on a fixed order of operations (such
    as rematerialization.)
    The flatbuffer exporter will unwrap the wrapped region and annotate the generated
    model with metadata such that any runtime reorderings will respect the order
    given by the control dependencies.
  }];

  // May take values of type !tfl.control as inputs
  let arguments = (ins
    Variadic<TFL_Control>:$controlInputs
  );

  // Adds a single value of type !tfl.control as the last output (all other
  // outputs are copied from the wrapped region.)
  let results = (outs
    Variadic<AnyTensor>:$outputs,
    TFL_Control:$control
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    YieldOp GetYield();
    bool WrapsSinglePerfectlyForwardedOp();
  }];

  let hasCanonicalizer = 0;
  let hasVerifier = 1;
  let hasFolder = 0;
  let hasCustomAssemblyFormat = 1;
}

#endif // TFL_OPS

// LINT.ThenChange()
// LINT.ThenChange(//tensorflow/lite/tools/versioning/op_version.cc)
