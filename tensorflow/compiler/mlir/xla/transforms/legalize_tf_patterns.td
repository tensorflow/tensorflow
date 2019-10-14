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

// This is the legalization pattern definition file for TF to XLA.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops.td"

def NullArrayAttr : NativeCodeCall<"ArrayAttr()">;
def NullDenseIntElementsAttr : NativeCodeCall<"DenseIntElementsAttr()">;

def CastIntElementsAttr : NativeCodeCall<"$0.cast<DenseIntElementsAttr>()">;

//===----------------------------------------------------------------------===//
// BatchNorm op patterns.
//===----------------------------------------------------------------------===//

def FeatureDimension : NativeCodeCall<
    "getFeatureDimensionAttr($_builder, $0, $1)">;
def FalseBoolAttr : AttrConstraint<CPred<"!$_self.getValue()">>;
def TrueBoolAttr : AttrConstraint<CPred<"$_self.getValue()">>;

def : Pattern<
    (TF_FusedBatchNormOp:$root $x, $scale, $offset, $mean, $variance, $epsilon,
                               $data_format, FalseBoolAttr:$is_training),
    [(HLO_BatchNormInferenceOp $x, $scale, $offset, $mean, $variance,
                               $epsilon, (FeatureDimension $data_format, $x)),
     // We already guaranteed that the last four results has no use so it
     // does not matter what value we provide here for replacement.
     /*batch_mean=*/(replaceWithValue $x),
     /*batch_variance=*/(replaceWithValue $x),
     /*reserve_space_1=*/(replaceWithValue $x),
     /*reserve_space_2=*/(replaceWithValue $x)],
    [(HasNoUseOf:$root__1), (HasNoUseOf:$root__2),
     (HasNoUseOf:$root__3), (HasNoUseOf:$root__4)]>;

//===----------------------------------------------------------------------===//
// Bias op patterns.
//===----------------------------------------------------------------------===//
def BiasAddFeatureDimension : NativeCodeCall<
    "getBiasFeatureDimension($_builder, $0, $1)">;

def : Pat<(TF_BiasAddOp AnyStaticShapeTensor:$input, $bias, $data_format),
          (HLO_AddOp $input, $bias,
              (BiasAddFeatureDimension $data_format, $input))>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

// Get the broadcast dimensions attribute from the binary operands.
def BinBroadcastDimensions : NativeCodeCall<
    "getBroadcastDimensionsAttr($_builder, $0, $1)">;

// Check that two values can be broadcasted together
def AreBroadcastCompatible : Constraint<CPred<"AreBroadcastCompatible($0, $1)">,
    "types must be broadcastable">;

class DirectBinaryPat<Op FromOp, Op ToOp>
  : Pat<(FromOp AnyTensor:$l, AnyTensor:$r),
        (ToOp $l, $r, (BinBroadcastDimensions $l, $r))>;

foreach fromToBinPair = [[TF_AddOp, HLO_AddOp],
                         [TF_AddV2Op, HLO_AddOp],
                         [TF_DivOp, HLO_DivOp],
                         [TF_MaximumOp, HLO_MaxOp],
                         [TF_MinimumOp, HLO_MinOp],
                         [TF_MulOp, HLO_MulOp],
                         [TF_RealDivOp, HLO_DivOp],
                         [TF_SubOp, HLO_SubOp]] in
  def : DirectBinaryPat<fromToBinPair[0], fromToBinPair[1]>;

//===----------------------------------------------------------------------===//
// Logical binary op patterns.
//===----------------------------------------------------------------------===//

class DirectLogicalBinaryPat<Op FromOp, Op ToOp>
  : Pat<(FromOp I1Tensor:$l, I1Tensor:$r),
        (ToOp $l, $r, (BinBroadcastDimensions $l, $r))>;

foreach fromToBinPair = [[TF_LogicalAndOp, HLO_AndOp],
                         [TF_LogicalOrOp, HLO_OrOp]] in
  def : DirectLogicalBinaryPat<fromToBinPair[0], fromToBinPair[1]>;

//===----------------------------------------------------------------------===//
// Compare op patterns.
//===----------------------------------------------------------------------===//

class DirectComparePat<Op FromOp, StrEnumAttrCase direction>
  : Pat<(FromOp AnyTensor:$l, AnyTensor:$r),
        (HLO_CompareOp $l, $r, (BinBroadcastDimensions $l, $r), direction)>;

def : DirectComparePat<TF_GreaterOp, HLO_COMPARISON_DIRECTION_GT>;
def : DirectComparePat<TF_GreaterEqualOp, HLO_COMPARISON_DIRECTION_GE>;
def : DirectComparePat<TF_LessOp, HLO_COMPARISON_DIRECTION_LT>;
def : DirectComparePat<TF_LessEqualOp, HLO_COMPARISON_DIRECTION_LE>;

class EqualityPat<Op FromOp, StrEnumAttrCase direction>
    : Pat<(FromOp AnyTensor:$l, AnyTensor:$r,
           TrueBoolAttr:$incompatible_shape_error),
        (HLO_CompareOp $l, $r, (BinBroadcastDimensions $l, $r), direction),
        [(AreBroadcastCompatible $l, $r)]>;

def : EqualityPat<TF_EqualOp, HLO_COMPARISON_DIRECTION_EQ>;
def : EqualityPat<TF_NotEqualOp, HLO_COMPARISON_DIRECTION_NE>;


//===----------------------------------------------------------------------===//
// Concat op patterns.
//===----------------------------------------------------------------------===//

def OneElementAttrPred
  : CPred<"$_self.cast<ElementsAttr>().getType().getNumElements() == 1">;

def OneElementAttr
  : ElementsAttrBase<And<[ElementsAttr.predicate, OneElementAttrPred]>,
                     "Scalar ElementsAttr">;

def GetHLOAxisFromTFAxis : NativeCodeCall<
  "GetHLOAxisFromTFAxis("
  "$0, (*$1.begin())->getType().cast<RankedTensorType>().getRank(), "
  "&$_builder)">;

def HasRankedFirstOperand
  : Constraint<CPred<"(*$0.begin())->getType().isa<RankedTensorType>()">>;

def IsShapedTensor
  : Constraint<CPred<"$0->getType().isa<RankedTensorType>()">>;

// This pattern converts TensorFlow axis format to HLO axis format which
// doesn't wrap around like TensorFlow and is always positive. For this
// conversion, use the first input to get inputs rank. Other inputs need not be
// ranked.
// Defining op for `axis` is TensorFlow constant op in the pattern as during
// the conversion, original Concat op operands still refers to the old ops even
// if HLO constant op is introduced as an replacement for the TensorFlow
// Constant op.
def : Pat<(TF_ConcatV2Op $inputs, (TF_ConstOp OneElementAttr:$axis), $unused),
          (HLO_ConcatenateOp $inputs, (GetHLOAxisFromTFAxis $axis, $inputs)),
          [(HasRankedFirstOperand $inputs)]>;

//===----------------------------------------------------------------------===//
// Identity op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_IdentityOp $op), (replaceWithValue $op)>;

//===----------------------------------------------------------------------===//
// Matmul op patterns.
//===----------------------------------------------------------------------===//

// TODO(hinsu): Lower matmul ops with transpose attributes.
def : Pat<(TF_MatMulOp $a, $b, ConstBoolAttrFalse, ConstBoolAttrFalse),
          (HLO_DotOp $a, $b, (NullArrayAttr))>;

//===----------------------------------------------------------------------===//
// Nullary op patterns.
//===----------------------------------------------------------------------===//

// TODO(riverriddle) Formalize a policy on converting opaque attributes.
def : Pat<(TF_ConstOp:$res ElementsAttr:$value), (HLO_ConstOp $value),
          [(AnyStaticShapeTensor $res)]>;

//===----------------------------------------------------------------------===//
// Relu op patterns.
//===----------------------------------------------------------------------===//

class ConstantSplat<string value> : NativeCodeCall<
    "getSplat($_builder, $0, " # value # ")">;

// TODO(hinsu): Support dynamic but ranked input by using scalar constants
// along with broadcast_dimensions attribute. Similarly, Relu6 op with
// non-static shape inputs can be lowered.
def : Pat<(TF_ReluOp AnyStaticShapeTensor:$input),
          (HLO_MaxOp (HLO_ConstOp (ConstantSplat<"0"> $input)), $input,
                     (NullDenseIntElementsAttr))>;

def : Pat<(TF_Relu6Op AnyStaticShapeTensor:$input),
          (HLO_ClampOp (HLO_ConstOp (ConstantSplat<"0"> $input)), $input,
                       (HLO_ConstOp (ConstantSplat<"6"> $input)))>;

// ReluGrad(gradients, features) = gradients * (features > 0)
def : Pat<(TF_ReluGradOp AnyStaticShapeTensor:$gradients, AnyStaticShapeTensor:$features),
          (HLO_SelectOp
            (HLO_CompareOp $features, (HLO_ConstOp:$zero (ConstantSplat<"0"> $features)),
              (NullDenseIntElementsAttr), HLO_COMPARISON_DIRECTION_GT),
            $gradients, $zero)>;

//===----------------------------------------------------------------------===//
// Relu op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_SelectOp AnyTensor:$condition, AnyTensor:$lhs, AnyTensor:$rhs),
          (HLO_SelectOp $condition, $lhs, $rhs)>;


//===----------------------------------------------------------------------===//
// Slice op patterns.
//===----------------------------------------------------------------------===//

def CanBeTranslatedToDynamicSlice : Constraint<CPred<
  "CanBeTranslatedToDynamicSlice($0, $1, $2.cast<DenseIntElementsAttr>())">>;

def TFSliceSizes2HLOSliceSizes : NativeCodeCall<
    "TFSliceSizes2HLOSliceSizes($0, $1, $2.cast<DenseIntElementsAttr>(),"
    "&$_builder)">;

def : Pat<(TF_SliceOp HLO_Tensor:$input, HLO_Tensor:$starting_indices,
           (TF_ConstOp I64ElementsAttr:$slice_sizes)),
          (HLO_DynamicSliceOp $input, $starting_indices,
           (TFSliceSizes2HLOSliceSizes $input, $starting_indices, $slice_sizes)),
          [(CanBeTranslatedToDynamicSlice $input, $starting_indices,
            $slice_sizes)]>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

foreach Mapping = [
                   [TF_AbsOp, HLO_AbsOp],
                   [TF_CeilOp, HLO_CeilOp],
                   [TF_CosOp, HLO_CosOp],
                   [TF_ExpOp, HLO_ExpOp],
                   [TF_FloorOp, HLO_FloorOp],
                   [TF_LogOp, HLO_LogOp],
                   [TF_NegOp, HLO_NegOp],
                   [TF_RsqrtOp, HLO_RsqrtOp],
                   [TF_TanhOp, HLO_TanhOp],
                  ] in {
 def : Pat<(Mapping[0] HLO_Tensor:$input),
           (Mapping[1] $input)>;
}

// TODO(bixia): Lower Cast with a Complex type source operand or with
// Truncate=True for floating point value conversions.
def : Pat<(TF_CastOp HLO_Tensor:$arg, ConstBoolAttrFalse),
          (HLO_ConvertOp $arg)>;

def : Pat<(TF_TransposeOp:$res AnyStaticShapeTensor:$arg,
           (TF_ConstOp I64ElementsAttr:$permutation)),
          (HLO_TransposeOp $arg, (CastIntElementsAttr $permutation))>;

foreach TfOp = [TF_ExpandDimsOp, TF_ReshapeOp, TF_SqueezeOp, ] in {
  def : Pat<(TfOp:$res AnyStaticShapeTensor:$arg, $ignored),
            (HLO_ReshapeOp $arg), [(AnyStaticShapeTensor $res)]>;
}

//===----------------------------------------------------------------------===//
// RngUniform.
//===----------------------------------------------------------------------===//
def CastElementsToI64: NativeCodeCall<
  "CastElementsToI64($0->getLoc(), $1, &$_builder)">;

// TODO(misard,phawkins): handle random number generator seeds/states correctly.
def : Pat<(TF_RandomUniformOp:$old $shape, $seed, $seed2),
          (HLO_RngUniformOp
            (HLO_ConstOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.dtype(), 0.0)">)),
            (HLO_ConstOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.dtype(), 1.0)">)),
            (CastElementsToI64 $old, $shape)),
            [(IsShapedTensor $shape)]>;
