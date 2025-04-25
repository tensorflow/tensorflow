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
include "mlir/Dialect/Shape/IR/ShapeOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "stablehlo/dialect/ChloOps.td"
include "stablehlo/dialect/StablehloOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "mhlo/IR/hlo_ops.td"  // for hlo_utils.td

def SignedIntTensor : TensorOf<[I1, I8, I16, I32, I64]>;
def UnsignedIntTensor : TensorOf<[UI8, UI16, UI32, UI64]>;

// IEEE compliant floating point tensors.
def IEEEFloatTensor : TensorOf<[F16, F32, F64]>;

//===----------------------------------------------------------------------===//
// BatchNorm op patterns.
//===----------------------------------------------------------------------===//

def FalseBoolAttr : AttrConstraint<CPred<"!llvm::cast<BoolAttr>($_self).getValue()">>;
def TrueBoolAttr : AttrConstraint<CPred<"llvm::cast<BoolAttr>($_self).getValue()">>;

def CastValueToI64: NativeCodeCall<
  "CastValueToI64($0.getLoc(), $1, &$_builder)">;

def CastValueToElementType: NativeCodeCall<
  "$_builder.create<stablehlo::ConvertOp>($0.getLoc(), $1, "
  "getElementTypeOrSelf($2.getType()))">;

// Here, $0 is an ElementsAttr with exactly one element of type integer. $1 is
// the corresponding value of ranked tensor type whose axis is referred in $0.
def GetHLOAxisFromTFAxis : NativeCodeCall<
  "GetHLOAxisFromTFAxis("
  "$0, llvm::cast<RankedTensorType>($1.getType()).getRank(), &$_builder)">;

// Same as the above but with $1 of type operand_range from variadic TensorFlow
// input.
def GetHLOAxisFromTFAxisVariadic : NativeCodeCall<
  "GetHLOAxisFromTFAxis("
  "$0, llvm::cast<RankedTensorType>((*$1.begin()).getType()).getRank(), "
  "&$_builder)">;

def CastElementsToI64Elements : NativeCodeCall<[{
  llvm::cast<mlir::DenseIntElementsAttr>(hlo::convertElementsAttr(
    llvm::cast<ElementsAttr>($0), $_builder.getIntegerType(64)))
  }]>;

def CastElementsToI64Array : NativeCodeCall<[{
  ToDenseI64ArrayAttr(
    llvm::cast<mlir::DenseIntElementsAttr>(hlo::convertElementsAttr(
      llvm::cast<ElementsAttr>($0), $_builder.getIntegerType(64))), &$_builder)
  }]>;

def EmptyDotAlgorithmAttr : NativeCodeCall<"mlir::stablehlo::DotAlgorithmAttr{}">;

def ConstDefaultResultAccuracyAttr :
  ConstantAttr<StableHLO_ResultAccuracyAttr, "::mlir::stablehlo::ResultAccuracyMode::DEFAULT">;

//===----------------------------------------------------------------------===//
// ApproximateEqual op pattern.
//===----------------------------------------------------------------------===//

class StableHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<StableHLO_ComparisonDirectionAttr, "::mlir::stablehlo::ComparisonDirection::" # enumStr>;

class CHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<CHLO_ComparisonDirectionAttr, "::mlir::chlo::ComparisonDirection::" # enumStr>;

// TODO(b/228291745): Assert that $x and $y have the same shape.
def : Pat<(TF_ApproximateEqualOp:$result $x, $y, $tolerance),
          (CHLO_BroadcastCompareOp
           (StableHLO_AbsOp:$abs (StableHLO_SubtractOp $x, $y)),
           (CastValueToElementType $result, (StableHLO_ConstantOp $tolerance), $abs),
           (NullDenseI64ArrayAttr),
           CHLO_ComparisonDirectionValue<"LT">,
           (CHLO_DEFAULT_COMPARISON_TYPE))>;

//===----------------------------------------------------------------------===//
// Assert op pattern.
//===----------------------------------------------------------------------===//

// HLO and XLA doesn't support Assertions.
def LowerAssert : Pattern<(TF_AssertOp $condition, $data, $summarize), []>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

// Check that two values can be broadcasted together
def AreBroadcastCompatible : Constraint<CPred<"AreBroadcastCompatible($0, $1)">,
    "types must be broadcastable">;

class DirectBinaryPat<Op FromOp, Op ToOp>
  : Pat<(FromOp AnyTensor:$l, AnyTensor:$r),
        (ToOp $l, $r, (BinBroadcastDimensions $l, $r))>;

foreach fromToBinPair = [[TF_AddV2Op, CHLO_BroadcastAddOp],
                         [TF_Atan2Op, CHLO_BroadcastAtan2Op],
                         [TF_ComplexOp, CHLO_BroadcastComplexOp],
                         [TF_DivOp, CHLO_BroadcastDivOp],
                         [TF_LeftShiftOp, CHLO_BroadcastShiftLeftOp],
                         [TF_MaximumOp, CHLO_BroadcastMaxOp],
                         [TF_MinimumOp, CHLO_BroadcastMinOp],
                         [TF_ModOp, CHLO_BroadcastRemOp],
                         [TF_MulOp, CHLO_BroadcastMulOp],
                         [TF_NextAfterOp, CHLO_BroadcastNextAfterOp],
                         [TF_PolygammaOp, CHLO_BroadcastPolygammaOp],
                         [TF_PowOp, CHLO_BroadcastPowOp],
                         [TF_RealDivOp, CHLO_BroadcastDivOp],
                         [TF_SubOp, CHLO_BroadcastSubOp],
                         [TF_ZetaOp, CHLO_BroadcastZetaOp]] in
  def : DirectBinaryPat<fromToBinPair[0], fromToBinPair[1]>;

def LowerRightShiftSigned :
  Pat<(TF_RightShiftOp AnyTensor:$l, AnyTensor:$r),
      (CHLO_BroadcastShiftRightArithmeticOp $l, $r,
       (BinBroadcastDimensions $l, $r)),
      [(SignedIntTensor $r)]>;

def LowerRightShiftUnsigned :
  Pat<(TF_RightShiftOp AnyTensor:$l, AnyTensor:$r),
      (CHLO_BroadcastShiftRightLogicalOp $l, $r,
       (BinBroadcastDimensions $l, $r)),
      [(UnsignedIntTensor $r)]>;

// Performs a substitution of FloorDiv, pseudo code below:
//
//  return floor(div(x, y))
def : Pat<(TF_FloorDivOp AnyTensor:$l, AnyTensor:$r),
          (StableHLO_FloorOp
           (CHLO_BroadcastDivOp $l, $r, (BinBroadcastDimensions $l, $r))),
          [(IEEEFloatTensor $l)]>;

// Performs a substitution of FloorDiv for integer tensors, which required
// additional correction for a negative numerator / denominator. Equivalent
// pseudocode is shown below:
//
// T z = x / y
// return (z * y != x && (x < 0) != (y < 0)) ? z - 1 : z
//
// BroadcastToDimensions is used to compute the broadcast attr to higher
// dimensions. This computes the broadcast of 'l' to broadcast('l', 'r')
// without returning the broadcast of 'r' to broadcast('l', 'r').
def : Pat<(TF_FloorDivOp AnyTensor:$l, AnyTensor:$r),
      (StableHLO_SelectOp
       (CHLO_BroadcastAndOp
        (CHLO_BroadcastCompareOp
         (CHLO_BroadcastMulOp:$mul
          (CHLO_BroadcastDivOp:$div $l, $r,
           (BinBroadcastDimensions $l, $r)),
          $r, (BinBroadcastDimensions $div, $r)),
         $l, (BinBroadcastDimensions $mul, $l), CHLO_ComparisonDirectionValue<"NE">,
         (CHLO_DEFAULT_COMPARISON_TYPE)),
        (CHLO_BroadcastCompareOp
         (CHLO_BroadcastCompareOp:$l_cmp $l,
          (StableHLO_ConstantOp:$l_zeros (GetScalarOfType<0> $l)),
          (NullDenseI64ArrayAttr), CHLO_ComparisonDirectionValue<"LT">,
          (CHLO_DEFAULT_COMPARISON_TYPE)),
         (CHLO_BroadcastCompareOp:$r_cmp $r,
          (StableHLO_ConstantOp:$r_zeros (GetScalarOfType<0> $r)),
          (NullDenseI64ArrayAttr), CHLO_ComparisonDirectionValue<"LT">,
          (CHLO_DEFAULT_COMPARISON_TYPE)),
         (BinBroadcastDimensions $l_cmp, $r_cmp), CHLO_ComparisonDirectionValue<"NE">,
         (CHLO_DEFAULT_COMPARISON_TYPE)),
        (NullDenseI64ArrayAttr)),
       (CHLO_BroadcastSubOp $div,
        (StableHLO_ConstantOp:$ones (GetScalarOfType<1> $div)),
        (NullDenseI64ArrayAttr)), $div),
      [(SignedIntTensor $l)]>;

// FloorDiv of unsigned is just div.
def : Pat<(TF_FloorDivOp AnyTensor:$l, AnyTensor:$r),
          (CHLO_BroadcastDivOp $l, $r, (BinBroadcastDimensions $l, $r)),
          [(UnsignedIntTensor $l)]>;

// Performs a substitution of FloorMod designed to correct for possibly negative
// values. Pseudocode shown below:
//
//   T trunc_mod = std::fmod(x, y);
//   return trunc_mod != 0 && (y < 0 != trunc_mod < 0) ? trunc_mod + y
//                                                     : trunc_mod
def : Pat<(TF_FloorModOp AnyTensor:$l, AnyTensor:$r),
      (StableHLO_SelectOp
       (CHLO_BroadcastAndOp
        (CHLO_BroadcastCompareOp
         (CHLO_BroadcastRemOp:$rem $l, $r, (BinBroadcastDimensions $l, $r)),
         (StableHLO_ConstantOp:$l_zeros (GetScalarOfType<0> $l)),
         (NullDenseI64ArrayAttr), CHLO_ComparisonDirectionValue<"NE">,
         (CHLO_DEFAULT_COMPARISON_TYPE)),
        (CHLO_BroadcastCompareOp
         (CHLO_BroadcastCompareOp:$r_cmp $r,
          (StableHLO_ConstantOp:$r_zeros (GetScalarOfType<0> $r)),
          (NullDenseI64ArrayAttr), CHLO_ComparisonDirectionValue<"LT">,
          (CHLO_DEFAULT_COMPARISON_TYPE)),
         (CHLO_BroadcastCompareOp:$rem_cmp $rem, $r_zeros,
          (BinBroadcastDimensions $rem, $r_zeros), CHLO_ComparisonDirectionValue<"LT">,
          (CHLO_DEFAULT_COMPARISON_TYPE)),
         (BinBroadcastDimensions $r_cmp, $rem_cmp), CHLO_ComparisonDirectionValue<"NE">,
         (CHLO_DEFAULT_COMPARISON_TYPE)),
        (NullDenseI64ArrayAttr)),
       (CHLO_BroadcastAddOp $r,
        $rem, (BinBroadcastDimensions $r, $rem)), $rem),
      [(TensorOf<[I8, I16, I32, I64, F16, F32, F64]> $l)]>;

// FloorMod of unsigned is just mod.
def : Pat<(TF_FloorModOp AnyTensor:$l, AnyTensor:$r),
          (CHLO_BroadcastRemOp $l, $r, (BinBroadcastDimensions $l, $r)),
          [(UnsignedIntTensor $l)]>;

def Get2DTransposePerm: NativeCodeCall<
  "Get2DTransposePerm($0, &$_builder)">;

def : Pat<(TF_RiscAddOp $l, $r), (StableHLO_AddOp $l, $r)>;

def : Pat<(TF_RiscDotOp $a, $b, $transpose_a, $transpose_b),
          (StableHLO_DotOp
          (TF_TransposeOp $a, (TF_ConstOp (Get2DTransposePerm $transpose_a))),
          (TF_TransposeOp $b, (TF_ConstOp (Get2DTransposePerm $transpose_b))),
          /*precision_config=*/(NullArrayAttr))>;

//===----------------------------------------------------------------------===//
// Logical & bitwise binary op patterns.
//===----------------------------------------------------------------------===//

class DirectLogicalBinaryPat<Op FromOp, Op ToOp>
  : Pat<(FromOp AnyTensor:$l, AnyTensor:$r),
        (ToOp $l, $r, (BinBroadcastDimensions $l, $r)),
        [(AnyTypeOf<[SignedIntTensor, UnsignedIntTensor]> $l)]>;

foreach fromToBinPair = [[TF_LogicalAndOp, CHLO_BroadcastAndOp],
                         [TF_LogicalOrOp, CHLO_BroadcastOrOp],
                         [TF_BitwiseAndOp, CHLO_BroadcastAndOp],
                         [TF_BitwiseOrOp, CHLO_BroadcastOrOp],
                         [TF_BitwiseXorOp, CHLO_BroadcastXorOp]] in
  def : DirectLogicalBinaryPat<fromToBinPair[0], fromToBinPair[1]>;

//===----------------------------------------------------------------------===//
// Compare op patterns.
//===----------------------------------------------------------------------===//

class DirectComparePat<Op FromOp, CHLO_ComparisonDirectionValue direction>
  : Pat<(FromOp AnyTensor:$l, AnyTensor:$r),
        (CHLO_BroadcastCompareOp
           $l, $r, (BinBroadcastDimensions $l, $r), direction,
           (CHLO_DEFAULT_COMPARISON_TYPE))>;

def : DirectComparePat<TF_GreaterOp, CHLO_ComparisonDirectionValue<"GT">>;
def : DirectComparePat<TF_GreaterEqualOp, CHLO_ComparisonDirectionValue<"GE">>;
def : DirectComparePat<TF_LessOp, CHLO_ComparisonDirectionValue<"LT">>;
def : DirectComparePat<TF_LessEqualOp, CHLO_ComparisonDirectionValue<"LE">>;

class EqualityPat<Op FromOp, CHLO_ComparisonDirectionValue direction>
    : Pat<(FromOp AnyTensor:$l, AnyTensor:$r,
           TrueBoolAttr:$incompatible_shape_error),
        (CHLO_BroadcastCompareOp
         $l, $r, (BinBroadcastDimensions $l, $r), direction,
         (CHLO_DEFAULT_COMPARISON_TYPE)),
        [(HLO_Tensor $l)]>;

def : EqualityPat<TF_EqualOp, CHLO_ComparisonDirectionValue<"EQ">>;
def : EqualityPat<TF_NotEqualOp, CHLO_ComparisonDirectionValue<"NE">>;

//===----------------------------------------------------------------------===//
// Concat op patterns.
//===----------------------------------------------------------------------===//

def OneElementAttrPred
  : CPred<"llvm::cast<ElementsAttr>($_self).getShapedType().getNumElements() == 1">;

def OneElementAttr
  : ElementsAttrBase<And<[ElementsAttr.predicate, OneElementAttrPred]>,
                     "Scalar ElementsAttr">;

def HasRankedFirstOperand
  : Constraint<CPred<"llvm::isa<RankedTensorType>((*$0.begin()).getType())">>;

def IsShapedTensor
  : Constraint<CPred<"llvm::isa<RankedTensorType>($0.getType())">>;

// This pattern converts TensorFlow axis format to HLO axis format which
// doesn't wrap around like TensorFlow and is always positive. For this
// conversion, use the first input to get inputs rank. Other inputs need not be
// ranked.
// Defining op for `axis` is TensorFlow constant op in the pattern as during
// the conversion, original Concat op operands still refers to the old ops even
// if HLO constant op is introduced as an replacement for the TensorFlow
// Constant op.
def : Pat<(TF_ConcatV2Op $inputs, (ConstantLikeMatcher OneElementAttr:$axis)),
          (StableHLO_ConcatenateOp $inputs,
            (GetHLOAxisFromTFAxisVariadic $axis, $inputs)),
          [(HasRankedFirstOperand $inputs)]>;

//===----------------------------------------------------------------------===//
// CollectivePermute op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_CollectivePermuteOp $input, (ConstantLikeMatcher ElementsAttr:$source_target_pairs)),
          (StableHLO_CollectivePermuteOp $input,
            (CastElementsToI64Elements $source_target_pairs),
            (StableHLO_NullChannelHandleAttr))>;

//===----------------------------------------------------------------------===//
// CrossReplicaSum op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_CrossReplicaSumOp $input, (ConstantLikeMatcher ElementsAttr:$group_assignment)),
          (StableHLO_CrossReplicaSumOp $input,
            (CastElementsToI64Elements $group_assignment))>;

//===----------------------------------------------------------------------===//
// All2All op patterns.
//===----------------------------------------------------------------------===//

def ValueToVariadic: NativeCodeCall<"SmallVector<Value, 1>{$0}">;
def : Pat<(TF_AllToAllOp AnyRankedTensor:$input, (ConstantLikeMatcher ElementsAttr:$group_assignment), I64Attr:$concat_dimension, $split_dimension, $split_count),
          (StableHLO_AllToAllOp (ValueToVariadic $input), $split_dimension, $concat_dimension, $split_count, (CastElementsToI64Elements $group_assignment), (StableHLO_NullChannelHandleAttr))>;

//===----------------------------------------------------------------------===//
// FFT op patterns.
//===----------------------------------------------------------------------===//

class StableHLO_FftTypeValue<string enumStr> :
  ConstantAttr<StableHLO_FftTypeAttr, "::mlir::stablehlo::FftType::" # enumStr>;

def GetInnerDimFromValue : NativeCodeCall<
  "GetInnerDimFromValue(llvm::cast<ShapedType>($0.getType()), &$_builder)">;

def CheckInnerDimStatic
  : Constraint<CPred<"CheckInnerDimStatic(llvm::cast<ShapedType>($0.getType()), &$_builder)">>;

def : Pat<(TF_FFTOp:$res $input),
          (StableHLO_FftOp $input, StableHLO_FftTypeValue<"FFT">, (GetInnerDimFromValue $res)),
          [(CheckInnerDimStatic $input)]>;

def : Pat<(TF_IFFTOp:$res $input),
          (StableHLO_FftOp $input, StableHLO_FftTypeValue<"IFFT">, (GetInnerDimFromValue $res)),
          [(CheckInnerDimStatic $input)]>;

//===----------------------------------------------------------------------===//
// GatherV2 op patterns.
//===----------------------------------------------------------------------===//

// Here, $params and $indices needs to be ranked so that $axis and $batch_dims
// attributes can be converted from TensorFlow axis format supporting negative
// indexing to the HLO format.
def LegalizeGatherV2 :
  Pat<(TF_GatherV2Op AnyRankedTensor:$params, AnyRankedTensor:$indices,
        (ConstantLikeMatcher ElementsAttr:$axis), $batch_dims),
      (StableHLO_TorchIndexSelectOp $params, $indices,
        (GetHLOAxisFromTFAxis $axis, $params),
        (GetHLOAxisFromTFAxis $batch_dims, $indices))>;

//===----------------------------------------------------------------------===//
// Pad op patterns.
//===----------------------------------------------------------------------===//

class SliceDenseIntElementsAttrColumn2D<string column> : NativeCodeCall<
  "SliceDenseIntElementsAttrColumn2D(llvm::cast<ElementsAttr>($0), " # column # " )">;

class SliceDenseIntElementsAttr<string index, string axis> : NativeCodeCall<
  "SliceDenseIntElementsAttr(llvm::cast<ElementsAttr>($0), " # index # ", " # axis # ")">;

// Interior padding attribute based on the TF padding.
def GetInteriorPadding : NativeCodeCall<
  "GetInteriorPadding(llvm::cast<ElementsAttr>($0))">;

def : Pat<(TF_PadV2Op $input, (ConstantLikeMatcher ElementsAttr:$padding), $c),
          (StableHLO_PadOp $input, $c,
           (SliceDenseIntElementsAttrColumn2D<"0"> $padding),
           (SliceDenseIntElementsAttrColumn2D<"1"> $padding),
           (GetInteriorPadding $padding))>;

//===----------------------------------------------------------------------===//
// Identity op patterns.
//===----------------------------------------------------------------------===//

foreach src = [TF_IdentityOp, TF_StopGradientOp, TF__EagerConstOp] in
  def : Pat<(src $op), (replaceWithValue $op)>;

// TODO(b/32223192): Support CheckNumerics in HLO.
foreach src = [TF_PreventGradientOp, TF_CheckNumericsOp] in
  def : Pat<(src $op, $msg), (replaceWithValue $op)>;

//===----------------------------------------------------------------------===//
// MatMul op patterns.
//===----------------------------------------------------------------------===//

def StableHLO_GetPrecisionConfig: NativeCodeCall<
  "GetPrecisionConfig(&$_builder)">;

def : Pat<(TF_MatMulOp $a, $b, $transpose_a, $transpose_b, $grad_a, $grad_b),
          (StableHLO_DotOp
          (TF_TransposeOp $a, (TF_ConstOp (Get2DTransposePerm $transpose_a))),
          (TF_TransposeOp $b, (TF_ConstOp (Get2DTransposePerm $transpose_b))),
          /*precision_config=*/(StableHLO_GetPrecisionConfig))>;

//===----------------------------------------------------------------------===//
// Lower `tf.ZerosLike`
//===----------------------------------------------------------------------===//

def : Pat<(TF_ZerosLikeOp AnyTensor:$arg),
          (StableHLO_ConstantLike<"0"> $arg)>;

//===----------------------------------------------------------------------===//
// Lower `tf.OnesLike`
//===----------------------------------------------------------------------===//

def : Pat<(TF_OnesLikeOp AnyTensor:$arg),
          (StableHLO_ConstantLike<"1"> $arg)>;

//===----------------------------------------------------------------------===//
// Elu op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_EluOp AnyTensor:$features),
          (StableHLO_SelectOp
           (StableHLO_CompareOp
              $features,
              (StableHLO_ConstantLike<"0">:$zero $features),
              StableHLO_ComparisonDirectionValue<"GT">, (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
           $features,
           (StableHLO_Expm1Op $features, ConstDefaultResultAccuracyAttr))>;

def : Pat<(TF_EluGradOp AnyStaticShapeTensor:$gradients, AnyRankedTensor:$features),
           (StableHLO_SelectOp
            (CHLO_BroadcastCompareOp
              $features,
              (StableHLO_ConstantOp:$zero (GetScalarOfType<0> $features)),
              (BinBroadcastDimensions $zero, $features),
              CHLO_ComparisonDirectionValue<"GT">, (CHLO_DEFAULT_COMPARISON_TYPE)),
            $gradients,
            (StableHLO_MulOp
             $gradients,
             (CHLO_BroadcastAddOp
               $features,
               (StableHLO_ConstantOp:$one (GetScalarOfType<1> $features)),
               (BinBroadcastDimensions $one, $features))))>;

//===----------------------------------------------------------------------===//
// Relu op patterns.
//===----------------------------------------------------------------------===//

// TODO(hinsu): Make these patterns to TF to TF lowering. Relu6 lowering will
// require HLO canonicalization of min and max on a tensor to ClampOp.

// TODO(hinsu): Lower quantized types after supporting them in GetScalarOfType.
def : Pat<(TF_ReluOp AnyTensor:$input),
          (CHLO_BroadcastMaxOp
               (StableHLO_ConstantOp:$zero (GetScalarOfType<0> $input)), $input,
               (BinBroadcastDimensions $zero, $input)),
          [(TF_IntOrFpTensor $input)]>;

// TODO(hinsu): Lower quantized types after supporting them in GetScalarOfType.
def : Pat<(TF_Relu6Op AnyRankedTensor:$input),
          (StableHLO_ClampOp (StableHLO_ConstantOp (GetScalarOfType<0> $input)), $input,
                       (StableHLO_ConstantOp (GetScalarOfType<6> $input))),
          [(TF_IntOrFpTensor $input)]>;

// ReluGrad(gradients, features) = gradients * (features > 0)
// The condition that $gradients and $features need to have the same shape is
// implicitly enforced: $zero is created to have the same shape as $features,
// StableHLO_SelectOp enforces that $gradients and $zero have the same shape.
def : Pat<(TF_ReluGradOp AnyTensor:$gradients, AnyTensor:$features),
          (StableHLO_SelectOp
            (StableHLO_CompareOp $features, (StableHLO_ConstantLike<"0">:$zero $features),
              StableHLO_ComparisonDirectionValue<"GT">, (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            $gradients, $zero)>;

//===----------------------------------------------------------------------===//
// Softsign op patterns.
//===----------------------------------------------------------------------===//

/// Converts a TF::SoftsignOp to HLO.
/// Softsign(features) = features / (1 + abs(features))
def : Pat<(TF_SoftsignOp AnyTensor:$input),
          (StableHLO_DivOp
            $input,
            (StableHLO_AddOp (StableHLO_ConstantLike<"1"> $input), (StableHLO_AbsOp $input))
          )
         >;

/// Converts a TF::SoftsignGradOp to HLO.
/// SoftsignGrad(gradient, features) = gradient / ((1 + abs(features)) ^ 2)
def : Pattern<
        (TF_SoftsignGradOp AnyRankedTensor:$gradients, AnyRankedTensor:$features),
        [(CHLO_BroadcastAddOp:$add
          (StableHLO_ConstantOp:$one (GetScalarOfType<1> $features)), (StableHLO_AbsOp $features),
          (BinBroadcastDimensions $one, $features)
         ),
         (CHLO_BroadcastDivOp
           $gradients,
           (StableHLO_MulOp $add, $add),
           (BinBroadcastDimensions $gradients, $add)
         )
        ]>;

//===----------------------------------------------------------------------===//
// Slice op patterns.
//===----------------------------------------------------------------------===//

def UnpackStartingIndices: NativeCodeCall<
  "UnpackTensorAlongZeroDim($0.getLoc(), $1, &$_builder).getOutput()">;

def CanBeTranslatedToDynamicSlice : Constraint<CPred<
  "CanBeTranslatedToDynamicSlice($0, $1, llvm::cast<DenseIntElementsAttr>($2))">>;

def TFSliceSizes2HLOSliceSizes : NativeCodeCall<
    "TFSliceSizes2HLOSliceSizes($0, $1, llvm::cast<DenseIntElementsAttr>($2),"
    "&$_builder)">;

def : Pat<(TF_SliceOp:$op HLO_Tensor:$input, HLO_Tensor:$starting_indices,
           (ConstantLikeMatcher AnyAttr:$slice_sizes)),
          (StableHLO_DynamicSliceOp $input,
           (UnpackStartingIndices $op, $starting_indices),
           (TFSliceSizes2HLOSliceSizes $input, $starting_indices, $slice_sizes)),
          [(CanBeTranslatedToDynamicSlice $input, $starting_indices,
            $slice_sizes)]>;

//===----------------------------------------------------------------------===//
// Select op patterns.
//===----------------------------------------------------------------------===//

 def : Pat<(TF_SelectV2Op HLO_Tensor:$pred, HLO_Tensor:$on_true,
            HLO_Tensor:$on_false),
           (CHLO_BroadcastSelectOp $pred, $on_true, $on_false)>;

//===----------------------------------------------------------------------===//
// PartitionedCall and LegacyCall op patterns.
//===----------------------------------------------------------------------===//

def ArgTypesMatchCallee : Constraint<
    // $0 is a resultset (possibly empty), and $_op isn't assigned. So retrieve
    // the op using the builder.
    CPred<"ArgTypesMatchCallee(&*$_builder.getInsertionPoint(), $1, $2)">>;

foreach callOp = [TF_PartitionedCallOp, TF_StatefulPartitionedCallOp] in {
  def : Pat<(callOp:$op $args,  $args_attrs, $res_attrs, FlatSymbolRefAttr:$f,
             $config, $config_proto, $executor_type),
            (CallOp $f, $args,  $args_attrs, $res_attrs,
             ConstantAttr<UnitAttr, "false">),
          [(ArgTypesMatchCallee $op, $args, $f)]>;
}

// The extra attr on this op is _disable_call_shape_inference, which we ignore
// in the bridge.
def : Pat<(TF_LegacyCallOp:$op $args, $args_attrs, $res_attrs,
           FlatSymbolRefAttr:$f, $attr),
          (CallOp $f, $args, $args_attrs, $res_attrs,
           ConstantAttr<UnitAttr, "false">),
        [(ArgTypesMatchCallee $op, $args, $f)]>;

//===----------------------------------------------------------------------===//
// Reverse op patterns.
//===----------------------------------------------------------------------===//

// Handles axis conversion for TF reverse.
def ConvertAxisAttr : NativeCodeCall<"ConvertAxisAttr($0, llvm::cast<ElementsAttr>($1), &$_builder)">;

def : Pat<(TF_ReverseV2Op AnyRankedTensor:$values, (ConstantLikeMatcher ElementsAttr:$axis)),
    (StableHLO_ReverseOp $values, (ConvertAxisAttr $values, $axis))>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

foreach Mapping = [
                   [TF_AbsOp, StableHLO_AbsOp],
                   [TF_CeilOp, StableHLO_CeilOp],
                   [TF_ComplexAbsOp, StableHLO_AbsOp],
                   [TF_ErfOp, CHLO_ErfOp],
                   [TF_FloorOp, StableHLO_FloorOp],
                   [TF_ImagOp, StableHLO_ImagOp],
                   [TF_InvertOp, StableHLO_NotOp],
                   [TF_IsFiniteOp, StableHLO_IsFiniteOp],
                   [TF_LogicalNotOp, StableHLO_NotOp],
                   [TF_NegOp, StableHLO_NegOp],
                   [TF_RealOp, StableHLO_RealOp],
                  ] in {
 def : Pat<(Mapping[0] HLO_Tensor:$input),
           (Mapping[1] $input)>;
}


foreach Mapping = [
                   [TF_CosOp, StableHLO_CosineOp],
                   [TF_ExpOp, StableHLO_ExpOp],
                   [TF_Expm1Op, StableHLO_Expm1Op],
                   [TF_LogOp, StableHLO_LogOp],
                   [TF_Log1pOp, StableHLO_Log1pOp],
                   [TF_RsqrtOp, StableHLO_RsqrtOp],
                   [TF_SigmoidOp, StableHLO_LogisticOp],
                   [TF_SinOp, StableHLO_SineOp],
                   [TF_SqrtOp, StableHLO_SqrtOp],
                   [TF_TanhOp, StableHLO_TanhOp],
                   [TF_TanOp, StableHLO_TanOp]
                  ] in {
 def : Pat<(Mapping[0] HLO_Tensor:$input),
           (Mapping[1] $input, ConstDefaultResultAccuracyAttr)>;
}

foreach Mapping = [
                   [TF_AcosOp, CHLO_AcosOp],
                   [TF_AcoshOp, CHLO_AcoshOp],
                   [TF_AsinOp, CHLO_AsinOp],
                   [TF_AsinhOp, CHLO_AsinhOp],
                   [TF_AtanOp, CHLO_AtanOp],
                   [TF_AtanhOp, CHLO_AtanhOp],
                   [TF_CoshOp, CHLO_CoshOp],
                   [TF_ConjOp, CHLO_ConjOp],
                   [TF_DigammaOp, CHLO_DigammaOp],
                   [TF_ErfcOp, CHLO_ErfcOp],
                   [TF_IsInfOp, CHLO_IsInfOp],
                   [TF_LgammaOp, CHLO_LgammaOp],
                   [TF_SinhOp, CHLO_SinhOp],
                  ] in {
 def : Pat<(Mapping[0] HLO_AnyTensor:$input),
           (Mapping[1] $input)>;
}

def : Pat<(TF_AngleOp $x), (StableHLO_Atan2Op (StableHLO_ImagOp $x), (StableHLO_RealOp $x))>;

// TODO(bixia): Lower with Truncate=True for floating point value conversions.
def : Pat<(TF_CastOp $arg, ConstBoolAttrFalse), (StableHLO_ConvertOp $arg)>;

def : Pat<(TF_TransposeOp:$res $arg, (ConstantLikeMatcher ElementsAttr:$permutation)),
          (StableHLO_TransposeOp $arg, (CastElementsToI64Array $permutation))>;


// Lowering these ops with static shape to stablehlo.reshape
foreach TfOp = [TF_ExpandDimsOp, TF_ReshapeOp, TF_SqueezeOp, ] in {
  def : Pat<(TfOp:$res HLO_Tensor:$arg, $ignored),
            (StableHLO_ReshapeOp $arg), [(AnyStaticShapeTensor $res)], [],
            (addBenefit 2)>;
}

// Returns NaN if x is NaN, 0 if x is 0, -1 if x < 0 and 1 if x > 0.
def : Pat<(TF_SignOp $x), (StableHLO_SignOp $x)>;

def BothElementTypesSameWidthIntOrFloat : Constraint<CPred<
  "getElementTypeOrSelf($0.getType()).isIntOrFloat() && "
  "getElementTypeOrSelf($1.getType()).isIntOrFloat()">,
  "element types must be integers or floats">;

// TODO(mgester): Due to restrictions of xla::BitcastConvertType we currently
// only lower if both input and output types are int or float and have same width

def : Pat<(TF_BitcastOp:$res HLO_Tensor:$arg),
          (StableHLO_BitcastConvertOp $arg),
          [(BothElementTypesSameWidthIntOrFloat $res, $arg)]>;

// TODO(jpienaar): Lower constant like to constant to broadcast if dynamic
// and going to StableHLO.

//===----------------------------------------------------------------------===//
// Random ops.
//===----------------------------------------------------------------------===//
// TODO(b/148269299): handle random number generator seeds/states correctly.

class StableHLO_RngDistributionValue<string enumStr> :
  ConstantAttr<StableHLO_RngDistributionAttr, "::mlir::stablehlo::RngDistribution::" # enumStr>;

def : Pat<(TF_RandomUniformOp:$old $shape, $seed, $seed2),
          (StableHLO_RngOp
            (StableHLO_ConstantOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.getDtype(), 0.0)">)),
            (StableHLO_ConstantOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.getDtype(), 1.0)">)),
            (CastValueToI64 $old, $shape),
            StableHLO_RngDistributionValue<"UNIFORM">),
          [(IsShapedTensor $shape)]>;

def : Pat<(TF_RandomStandardNormalOp:$old $shape, $seed, $seed2),
          (StableHLO_RngOp
            (StableHLO_ConstantOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.getDtype(), 0.0)">)),
            (StableHLO_ConstantOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.getDtype(), 1.0)">)),
            (CastValueToI64 $old, $shape),
            StableHLO_RngDistributionValue<"NORMAL">),
          [(IsShapedTensor $shape)]>;

//===----------------------------------------------------------------------===//
// Sigmoid grad op.
//===----------------------------------------------------------------------===//

// Only handle static shape here, dynamic shape is handled by
// ConvertSigmoidGradOpDynamic
def HasStaticShape : Constraint<
    CPred<"::llvm::dyn_cast<ShapedType>($0.getType()).hasStaticShape()">>;

def : Pat<(TF_SigmoidGradOp AnyRankedTensor:$l, AnyRankedTensor:$r),
          (StableHLO_MulOp
           (StableHLO_MulOp $r, $l),
           (StableHLO_SubtractOp (StableHLO_ConstantOp (ConstantSplat<"1"> $l)), $l)),
          [(HasStaticShape $l)]>;

//===----------------------------------------------------------------------===//
// Softplus op.
//===----------------------------------------------------------------------===//

def EpsilonValue : NativeCodeCall<"GetEpsilonValue($0.getType())">;

def : Pattern<(TF_SoftplusOp AnyTensor:$features),
              [
                (StableHLO_ExpOp:$features_exp $features, ConstDefaultResultAccuracyAttr),
                (CHLO_BroadcastAddOp:$threshold
                 (StableHLO_LogOp (StableHLO_ConstantOp (EpsilonValue $features)), ConstDefaultResultAccuracyAttr),
                 (StableHLO_ConstantOp (GetScalarOfType<2> $features)),
                 (NullDenseI64ArrayAttr)
                ),
                (StableHLO_SelectOp:$output
                 (CHLO_BroadcastCompareOp
                  $features,
                  (StableHLO_NegOp $threshold),
                  (NullDenseI64ArrayAttr),
                  CHLO_ComparisonDirectionValue<"GT">,
                  (CHLO_DEFAULT_COMPARISON_TYPE)
                 ),
                 $features,
                 (StableHLO_SelectOp
                  (CHLO_BroadcastCompareOp
                   $features,
                   $threshold,
                   (NullDenseI64ArrayAttr),
                   CHLO_ComparisonDirectionValue<"LT">,
                   (CHLO_DEFAULT_COMPARISON_TYPE)
                  ),
                  $features_exp,
                  (StableHLO_Log1pOp $features_exp, ConstDefaultResultAccuracyAttr)
                 )
                ),
                (replaceWithValue $output)
              ]>;

//===----------------------------------------------------------------------===//
// XlaReplicaId op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaReplicaIdOp),
          (TF_CastOp (StableHLO_ReplicaIdOp), /*truncate=*/ConstBoolAttrFalse)>;

//===----------------------------------------------------------------------===//
// XlaGather op.
//===----------------------------------------------------------------------===//

def ToGatherDimNumsAttr : NativeCodeCall<"GetGatherDimNumsAttr($0, &$_builder)">;

def HasValidGatherDims : Constraint<CPred<"HasValidGatherDims($0)">>;

def : Pat<(TF_XlaGatherOp $operand, $start_indices, (ConstantLikeMatcher ElementsAttr:$slice_sizes),
                          $dimension_numbers, $indices_are_sorted),
          (StableHLO_GatherOp $operand, $start_indices,
                        (ToGatherDimNumsAttr $dimension_numbers),
                        (CastElementsToI64Array $slice_sizes),
                        $indices_are_sorted),
          [(HasValidGatherDims $dimension_numbers)]>;

//===----------------------------------------------------------------------===//
// XlaDotOp op.
//===----------------------------------------------------------------------===//

def ToDotDimNumsAttr : NativeCodeCall<"GetDotDimNumsAttr($0, &$_builder)">;

def ToPrecisionConfigsAttr : NativeCodeCall<"GetPrecisionConfigAttr($0, &$_builder)">;

def HasValidDotDims : Constraint<CPred<"HasValidDotDims($0)">>;

def HasValidPrecisionConfig : Constraint<CPred<"HasValidPrecisionConfig($0)">>;

def : Pat<(TF_XlaDotOp $lhs, $rhs, $dimension_numbers, $precision_config),
          (StableHLO_DotGeneralOp $lhs, $rhs,
                            (ToDotDimNumsAttr $dimension_numbers),
                            (ToPrecisionConfigsAttr $precision_config),
                            (EmptyDotAlgorithmAttr)),
          [(HasValidDotDims $dimension_numbers), (HasValidPrecisionConfig $precision_config)]>;

//===----------------------------------------------------------------------===//
// XlaDotV2Op op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaDotV2Op $lhs, $rhs, $dimension_numbers, $precision_config),
          (StableHLO_DotGeneralOp $lhs, $rhs,
                            (ToDotDimNumsAttr $dimension_numbers),
                            (ToPrecisionConfigsAttr $precision_config),
                            (EmptyDotAlgorithmAttr)),
          [(HasValidDotDims $dimension_numbers), (HasValidPrecisionConfig $precision_config)]>;

//===----------------------------------------------------------------------===//
// XlaDynamicSlice op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaDynamicSliceOp:$op HLO_Tensor:$input, HLO_Tensor:$starting_indices,
           (ConstantLikeMatcher AnyAttr:$slice_sizes)),
          (StableHLO_DynamicSliceOp $input,
           (UnpackStartingIndices $op, $starting_indices),
           (TFSliceSizes2HLOSliceSizes $input, $starting_indices, $slice_sizes))>;

//===----------------------------------------------------------------------===//
// XlaEisumOp op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaEinsumOp $lhs, $rhs, $equation),
          (StableHLO_EinsumOp $lhs, $rhs, $equation)>;

//===----------------------------------------------------------------------===//
// XlaOptimizationBarrierOp op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaOptimizationBarrierOp $args),
          (StableHLO_OptimizationBarrierOp $args)>;
