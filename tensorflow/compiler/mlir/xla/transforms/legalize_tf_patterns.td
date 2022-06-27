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
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.td"
include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"

def SignedIntTensor : TensorOf<[I1, I8, I16, I32, I64]>;
def UnsignedIntTensor : TensorOf<[UI8, UI16, UI32, UI64]>;

// IEEE compliant floating point tensors.
def IEEEFloatTensor : TensorOf<[F16, F32, F64]>;

//===----------------------------------------------------------------------===//
// BatchNorm op patterns.
//===----------------------------------------------------------------------===//

def FalseBoolAttr : AttrConstraint<CPred<"!$_self.cast<BoolAttr>().getValue()">>;
def TrueBoolAttr : AttrConstraint<CPred<"$_self.cast<BoolAttr>().getValue()">>;

def CastValueToI64: NativeCodeCall<
  "CastValueToI64($0.getLoc(), $1, &$_builder)">;

def CastValueToElementType: NativeCodeCall<
  "$_builder.create<ConvertOp>($0.getLoc(), $1, "
  "getElementTypeOrSelf($2.getType()))">;

// Here, $0 is an ElementsAttr with exactly one element of type integer. $1 is
// the corresponding value of ranked tensor type whose axis is referred in $0.
def GetHLOAxisFromTFAxis : NativeCodeCall<
  "GetHLOAxisFromTFAxis("
  "$0, $1.getType().cast<RankedTensorType>().getRank(), &$_builder)">;

// Same as the above but with $1 of type operand_range from variadic TensorFlow
// input.
def GetHLOAxisFromTFAxisVariadic : NativeCodeCall<
  "GetHLOAxisFromTFAxis("
  "$0, (*$1.begin()).getType().cast<RankedTensorType>().getRank(), "
  "&$_builder)">;

def CastElementsToI64Elements : NativeCodeCall<
  "hlo::ConvertElementsAttr("
    "$0.cast<ElementsAttr>(), $_builder.getIntegerType(64)).cast<DenseIntElementsAttr>()">;

//===----------------------------------------------------------------------===//
// ApproximateEqual op pattern.
//===----------------------------------------------------------------------===//

class HLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<HLO_ComparisonDirectionAttr, "::mlir::mhlo::ComparisonDirection::" # enumStr>;

// TODO(b/228291745): Assert that $x and $y have the same shape.
def : Pat<(TF_ApproximateEqualOp:$result $x, $y, $tolerance),
          (CHLO_BroadcastCompareOp
           (HLO_AbsOp:$abs (HLO_SubOp $x, $y)),
           (CastValueToElementType $result, (HLO_ConstantOp $tolerance), $abs),
           (NullDenseIntElementsAttr),
           HLO_ComparisonDirectionValue<"LT">,
           (HLO_DEFAULT_COMPARISON_TYPE))>;

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
          (HLO_FloorOp
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
      (HLO_SelectOp
       (CHLO_BroadcastAndOp
        (CHLO_BroadcastCompareOp
         (CHLO_BroadcastMulOp:$mul
          (CHLO_BroadcastDivOp:$div $l, $r,
           (BinBroadcastDimensions $l, $r)),
          $r, (BinBroadcastDimensions $div, $r)),
         $l, (BinBroadcastDimensions $mul, $l), HLO_ComparisonDirectionValue<"NE">,
         (HLO_DEFAULT_COMPARISON_TYPE)),
        (CHLO_BroadcastCompareOp
         (CHLO_BroadcastCompareOp:$l_cmp $l,
          (HLO_ConstantOp:$l_zeros (GetScalarOfType<0> $l)),
          (NullDenseIntElementsAttr), HLO_ComparisonDirectionValue<"LT">,
          (HLO_DEFAULT_COMPARISON_TYPE)),
         (CHLO_BroadcastCompareOp:$r_cmp $r,
          (HLO_ConstantOp:$r_zeros (GetScalarOfType<0> $r)),
          (NullDenseIntElementsAttr), HLO_ComparisonDirectionValue<"LT">,
          (HLO_DEFAULT_COMPARISON_TYPE)),
         (BinBroadcastDimensions $l_cmp, $r_cmp), HLO_ComparisonDirectionValue<"NE">,
         (HLO_DEFAULT_COMPARISON_TYPE)),
        (NullDenseIntElementsAttr)),
       (CHLO_BroadcastSubOp $div,
        (HLO_ConstantOp:$ones (GetScalarOfType<1> $div)),
        (NullDenseIntElementsAttr)), $div),
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
      (HLO_SelectOp
       (CHLO_BroadcastAndOp
        (CHLO_BroadcastCompareOp
         (CHLO_BroadcastRemOp:$rem $l, $r, (BinBroadcastDimensions $l, $r)),
         (HLO_ConstantOp:$l_zeros (GetScalarOfType<0> $l)),
         (NullDenseIntElementsAttr), HLO_ComparisonDirectionValue<"NE">,
         (HLO_DEFAULT_COMPARISON_TYPE)),
        (CHLO_BroadcastCompareOp
         (CHLO_BroadcastCompareOp:$r_cmp $r,
          (HLO_ConstantOp:$r_zeros (GetScalarOfType<0> $r)),
          (NullDenseIntElementsAttr), HLO_ComparisonDirectionValue<"LT">,
          (HLO_DEFAULT_COMPARISON_TYPE)),
         (CHLO_BroadcastCompareOp:$rem_cmp $rem, $r_zeros,
          (BinBroadcastDimensions $rem, $r_zeros), HLO_ComparisonDirectionValue<"LT">,
          (HLO_DEFAULT_COMPARISON_TYPE)),
         (BinBroadcastDimensions $r_cmp, $rem_cmp), HLO_ComparisonDirectionValue<"NE">,
         (HLO_DEFAULT_COMPARISON_TYPE)),
        (NullDenseIntElementsAttr)),
       (CHLO_BroadcastAddOp $r,
        $rem, (BinBroadcastDimensions $r, $rem)), $rem),
      [(TensorOf<[I8, I16, I32, I64, F16, F32, F64]> $l)]>;

// FloorMod of unsigned is just mod.
def : Pat<(TF_FloorModOp AnyTensor:$l, AnyTensor:$r),
          (CHLO_BroadcastRemOp $l, $r, (BinBroadcastDimensions $l, $r)),
          [(UnsignedIntTensor $l)]>;

def Get2DTransposePerm: NativeCodeCall<
  "Get2DTransposePerm($0, &$_builder)">;

def : Pat<(TF_RiscAddOp $l, $r), (HLO_AddOp $l, $r)>;

def : Pat<(TF_RiscDotOp $a, $b, $transpose_a, $transpose_b),
          (HLO_DotOp
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

class DirectComparePat<Op FromOp, HLO_ComparisonDirectionValue direction>
  : Pat<(FromOp AnyTensor:$l, AnyTensor:$r),
        (CHLO_BroadcastCompareOp
           $l, $r, (BinBroadcastDimensions $l, $r), direction,
           (HLO_DEFAULT_COMPARISON_TYPE))>;

def : DirectComparePat<TF_GreaterOp, HLO_ComparisonDirectionValue<"GT">>;
def : DirectComparePat<TF_GreaterEqualOp, HLO_ComparisonDirectionValue<"GE">>;
def : DirectComparePat<TF_LessOp, HLO_ComparisonDirectionValue<"LT">>;
def : DirectComparePat<TF_LessEqualOp, HLO_ComparisonDirectionValue<"LE">>;

class EqualityPat<Op FromOp, HLO_ComparisonDirectionValue direction>
    : Pat<(FromOp AnyTensor:$l, AnyTensor:$r,
           TrueBoolAttr:$incompatible_shape_error),
        (CHLO_BroadcastCompareOp
         $l, $r, (BinBroadcastDimensions $l, $r), direction,
         (HLO_DEFAULT_COMPARISON_TYPE)),
        [(HLO_Tensor $l)]>;

def : EqualityPat<TF_EqualOp, HLO_ComparisonDirectionValue<"EQ">>;
def : EqualityPat<TF_NotEqualOp, HLO_ComparisonDirectionValue<"NE">>;

//===----------------------------------------------------------------------===//
// Concat op patterns.
//===----------------------------------------------------------------------===//

def OneElementAttrPred
  : CPred<"$_self.cast<ElementsAttr>().getType().getNumElements() == 1">;

def OneElementAttr
  : ElementsAttrBase<And<[ElementsAttr.predicate, OneElementAttrPred]>,
                     "Scalar ElementsAttr">;

def HasRankedFirstOperand
  : Constraint<CPred<"(*$0.begin()).getType().isa<RankedTensorType>()">>;

def IsShapedTensor
  : Constraint<CPred<"$0.getType().isa<RankedTensorType>()">>;

// This pattern converts TensorFlow axis format to HLO axis format which
// doesn't wrap around like TensorFlow and is always positive. For this
// conversion, use the first input to get inputs rank. Other inputs need not be
// ranked.
// Defining op for `axis` is TensorFlow constant op in the pattern as during
// the conversion, original Concat op operands still refers to the old ops even
// if HLO constant op is introduced as an replacement for the TensorFlow
// Constant op.
def : Pat<(TF_ConcatV2Op $inputs, (ConstantLikeMatcher OneElementAttr:$axis)),
          (HLO_ConcatenateOp $inputs,
            (GetHLOAxisFromTFAxisVariadic $axis, $inputs)),
          [(HasRankedFirstOperand $inputs)]>;

//===----------------------------------------------------------------------===//
// CollectivePermute op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_CollectivePermuteOp $input, (ConstantLikeMatcher ElementsAttr:$source_target_pairs)),
          (HLO_CollectivePermuteOp $input,
            (CastElementsToI64Elements $source_target_pairs))>;

//===----------------------------------------------------------------------===//
// CrossReplicaSum op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_CrossReplicaSumOp $input, (ConstantLikeMatcher ElementsAttr:$group_assignment)),
          (HLO_CrossReplicaSumOp $input,
            (CastElementsToI64Elements $group_assignment))>;

//===----------------------------------------------------------------------===//
// All2All op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_AllToAllOp AnyRankedTensor:$input, (ConstantLikeMatcher ElementsAttr:$group_assignment), I64Attr:$concat_dimension, $split_dimension, $split_count),
          (HLO_AllToAllOp $input, $split_dimension, $concat_dimension, $split_count, (CastElementsToI64Elements $group_assignment))>;

//===----------------------------------------------------------------------===//
// FFT op patterns.
//===----------------------------------------------------------------------===//

class HLO_FftTypeValue<string enumStr> :
  ConstantAttr<HLO_FftTypeAttr, "::mlir::mhlo::FftType::" # enumStr>;

def GetInnerDimFromValue : NativeCodeCall<
  "GetInnerDimFromValue($0.getType().cast<ShapedType>(), &$_builder)">;

def CheckInnerDimStatic
  : Constraint<CPred<"CheckInnerDimStatic($0.getType().cast<ShapedType>(), &$_builder)">>;

def : Pat<(TF_FFTOp:$res $input),
          (HLO_FftOp $input, HLO_FftTypeValue<"FFT">, (GetInnerDimFromValue $res)),
          [(CheckInnerDimStatic $input)]>;

def : Pat<(TF_IFFTOp:$res $input),
          (HLO_FftOp $input, HLO_FftTypeValue<"IFFT">, (GetInnerDimFromValue $res)),
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
      (HLO_TorchIndexSelectOp $params, $indices,
        (GetHLOAxisFromTFAxis $axis, $params),
        (GetHLOAxisFromTFAxis $batch_dims, $indices))>;

//===----------------------------------------------------------------------===//
// Pad op patterns.
//===----------------------------------------------------------------------===//

class SliceDenseIntElementsAttrColumn2D<string column> : NativeCodeCall<
  "SliceDenseIntElementsAttrColumn2D($0.cast<ElementsAttr>(), " # column # " )">;

class SliceDenseIntElementsAttr<string index, string axis> : NativeCodeCall<
  "SliceDenseIntElementsAttr($0.cast<ElementsAttr>(), " # index # ", " # axis # ")">;

// Interior padding attribute based on the TF padding.
def GetInteriorPadding : NativeCodeCall <
  "GetInteriorPadding($0.cast<ElementsAttr>())">;

def : Pat<(TF_PadV2Op $input, (ConstantLikeMatcher ElementsAttr:$padding), $c),
          (HLO_PadOp $input, $c,
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

def : Pat<(TF_MatMulOp $a, $b, $transpose_a, $transpose_b),
          (HLO_DotOp
          (TF_TransposeOp $a, (TF_ConstOp (Get2DTransposePerm $transpose_a))),
          (TF_TransposeOp $b, (TF_ConstOp (Get2DTransposePerm $transpose_b))),
          /*precision_config=*/(NullArrayAttr))>;

//===----------------------------------------------------------------------===//
// MatrixBandPart op pattern.
//===----------------------------------------------------------------------===//

class getIntegerAttr<string x>: NativeCodeCall<
  "$_builder.getI64IntegerAttr(" # x # ")">;

class GetDimensionSizeFromEnd<string dimFromEnd>: NativeCodeCall<
  "$_builder.getIntegerAttr(getElementTypeOrSelf($1.getType()), "
  "                         GetDimensionSizeFromEnd($0, " # dimFromEnd # "))"
  >;

// TODO(b/149615308): Enable IotaOp usage as a child operation in a pattern
// For now, this op needs to be created in C++ because the expected output type
// cannot be inferred.
class createIotaOp<string dim>: NativeCodeCall<
  "$_builder.create<mhlo::IotaOp>($0.getOwner()->getLoc(), "
  "Get2DTensorType($1, $2), $_builder.getI64IntegerAttr(" # dim # "))">;

// This op needs to be created in C++ because the generated Convert Op has no
// way to specify shape information as an input. In the MatrixBandPart op
// lowering, ConvertOp is not a root operation and the appropriate types cannot
// be inferred, so we construct it manually.
def createConvertOp: NativeCodeCall<
  "CreateConvertOp(&($_builder), $0.getOwner()->getLoc(), $1, $2)">;

// Performs a substitution of MatrixBandPartOp for XLA HLO ops. Pseudocode is
// shown below, given a tensor `input` with k dimensions [I, J, K, ..., M, N]
// and two integers, `num_lower` and `num_upper`:
//
//   iota_m = { M x N matrix with 0,1,...M along the M dimension }
//   iota_n = { M x N matrix with 0,1,...N along the N dimension }
//   num_lower_or_m = (num_lower < 0) ? m : num_lower
//   num_upper_or_n = (num_upper < 0) ? n : num_upper
//   offset = iota_m - iota_n
//   indicator = (-num_lower_or_m < offset) & (offset < num_upper_or_n)
//   zero_matrix = { [I, J, K,...M, N] zero matrix }
//   return (indicator ? input : zero_matrix)
//
// TODO(b/149961547): Support dynamic shaped `input` in MatrixBandPartOp.
def : Pattern<(TF_MatrixBandPartOp:$op AnyStaticShapeTensor:$input, $num_lower,
               $num_upper),
         [(HLO_ConstantOp:$m_dim (GetDimensionSizeFromEnd<"1"> $input, $num_lower)),
          (HLO_ConstantOp:$n_dim (GetDimensionSizeFromEnd<"0"> $input, $num_upper)),
          (HLO_SelectOp:$num_lower_or_m
           (HLO_CompareOp
            $num_lower, (HLO_ConstantOp:$zero (ConstantSplat<"0"> $num_lower)),
            HLO_ComparisonDirectionValue<"LT">, (HLO_DEFAULT_COMPARISON_TYPE)
           ),
           $m_dim,
           $num_lower
          ),
          (HLO_SelectOp:$num_upper_or_n
           (HLO_CompareOp
            $num_upper, $zero, HLO_ComparisonDirectionValue<"LT">,
            (HLO_DEFAULT_COMPARISON_TYPE)
           ),
           $n_dim,
           $num_upper
          ),
          (TF_SelectV2Op
           (HLO_AndOp
            (CHLO_BroadcastCompareOp
             (HLO_NegOp $num_lower_or_m),
             (HLO_SubOp:$offset
              (createIotaOp<"1"> $op, $input, $num_lower),
              (createIotaOp<"0"> $op, $input, $num_lower)
             ),
             (NullDenseIntElementsAttr), HLO_ComparisonDirectionValue<"LE">,
             (HLO_DEFAULT_COMPARISON_TYPE)
            ),
            (CHLO_BroadcastCompareOp $offset, $num_upper_or_n,
             (NullDenseIntElementsAttr), HLO_ComparisonDirectionValue<"LE">,
             (HLO_DEFAULT_COMPARISON_TYPE)
            )
           ),
           $input,
           (HLO_ConstantOp (ConstantSplat<"0"> $input))
          )]>;

//===----------------------------------------------------------------------===//
// Lower `tf.ZerosLike`
//===----------------------------------------------------------------------===//

def : Pat<(TF_ZerosLikeOp AnyTensor:$arg),
          (HLO_ConstantLike<"0"> $arg)>;

//===----------------------------------------------------------------------===//
// Lower `tf.OnesLike`
//===----------------------------------------------------------------------===//

def : Pat<(TF_OnesLikeOp AnyTensor:$arg),
          (HLO_ConstantLike<"1"> $arg)>;

//===----------------------------------------------------------------------===//
// Elu op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_EluOp AnyTensor:$features),
          (HLO_SelectOp
           (HLO_CompareOp
              $features,
              (HLO_ConstantLike<"0">:$zero $features),
              HLO_ComparisonDirectionValue<"GT">, (HLO_DEFAULT_COMPARISON_TYPE)),
           $features,
           (HLO_Expm1Op $features))>;

def : Pat<(TF_EluGradOp AnyStaticShapeTensor:$gradients, AnyRankedTensor:$features),
           (HLO_SelectOp
            (CHLO_BroadcastCompareOp
              $features,
              (HLO_ConstantOp:$zero (GetScalarOfType<0> $features)),
              (BinBroadcastDimensions $zero, $features),
              HLO_ComparisonDirectionValue<"GT">, (HLO_DEFAULT_COMPARISON_TYPE)),
            $gradients,
            (HLO_MulOp
             $gradients,
             (CHLO_BroadcastAddOp
               $features,
               (HLO_ConstantOp:$one (GetScalarOfType<1> $features)),
               (BinBroadcastDimensions $one, $features))))>;

//===----------------------------------------------------------------------===//
// Relu op patterns.
//===----------------------------------------------------------------------===//

// TODO(hinsu): Make these patterns to TF to TF lowering. Relu6 lowering will
// require HLO canonicalization of min and max on a tensor to ClampOp.

// TODO(hinsu): Lower quantized types after supporting them in GetScalarOfType.
def : Pat<(TF_ReluOp AnyTensor:$input),
          (CHLO_BroadcastMaxOp
               (HLO_ConstantOp:$zero (GetScalarOfType<0> $input)), $input,
               (BinBroadcastDimensions $zero, $input)),
          [(TF_IntOrFpTensor $input)]>;

// TODO(hinsu): Lower quantized types after supporting them in GetScalarOfType.
def : Pat<(TF_Relu6Op AnyRankedTensor:$input),
          (HLO_ClampOp (HLO_ConstantOp (GetScalarOfType<0> $input)), $input,
                       (HLO_ConstantOp (GetScalarOfType<6> $input))),
          [(TF_IntOrFpTensor $input)]>;

// ReluGrad(gradients, features) = gradients * (features > 0)
// The condition that $gradients and $features need to have the same shape is
// implicitly enforced: $zero is created to have the same shape as $features,
// HLO_SelectOp enforces that $gradients and $zero have the same shape.
def : Pat<(TF_ReluGradOp AnyTensor:$gradients, AnyTensor:$features),
          (HLO_SelectOp
            (HLO_CompareOp $features, (HLO_ConstantLike<"0">:$zero $features),
              HLO_ComparisonDirectionValue<"GT">, (HLO_DEFAULT_COMPARISON_TYPE)),
            $gradients, $zero)>;

//===----------------------------------------------------------------------===//
// Softsign op patterns.
//===----------------------------------------------------------------------===//

/// Converts a TF::SoftsignOp to HLO.
/// Softsign(features) = features / (1 + abs(features))
def : Pat<(TF_SoftsignOp AnyTensor:$input),
          (HLO_DivOp
            $input,
            (HLO_AddOp (HLO_ConstantLike<"1"> $input), (HLO_AbsOp $input))
          )
         >;

/// Converts a TF::SoftsignGradOp to HLO.
/// SoftsignGrad(gradient, features) = gradient / ((1 + abs(features)) ^ 2)
def : Pattern<
        (TF_SoftsignGradOp AnyRankedTensor:$gradients, AnyRankedTensor:$features),
        [(CHLO_BroadcastAddOp:$add
          (HLO_ConstantOp:$one (GetScalarOfType<1> $features)), (HLO_AbsOp $features),
          (BinBroadcastDimensions $one, $features)
         ),
         (CHLO_BroadcastDivOp
           $gradients,
           (HLO_MulOp $add, $add),
           (BinBroadcastDimensions $gradients, $add)
         )
        ]>;

//===----------------------------------------------------------------------===//
// Slice op patterns.
//===----------------------------------------------------------------------===//

def UnpackStartingIndices: NativeCodeCall<
  "UnpackTensorAlongZeroDim($0.getLoc(), $1, &$_builder).output()">;

def CanBeTranslatedToDynamicSlice : Constraint<CPred<
  "CanBeTranslatedToDynamicSlice($0, $1, $2.cast<DenseIntElementsAttr>())">>;

def TFSliceSizes2HLOSliceSizes : NativeCodeCall<
    "TFSliceSizes2HLOSliceSizes($0, $1, $2.cast<DenseIntElementsAttr>(),"
    "&$_builder)">;

def : Pat<(TF_SliceOp:$op HLO_Tensor:$input, HLO_Tensor:$starting_indices,
           (ConstantLikeMatcher AnyAttr:$slice_sizes)),
          (HLO_DynamicSliceOp $input,
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
    CPred<"ArgTypesMatchCallee($0[0].getOwner(), $1, $2)">>;

foreach callOp = [TF_PartitionedCallOp, TF_StatefulPartitionedCallOp] in {
  def : Pat<(callOp:$op $args, FlatSymbolRefAttr:$f,
             $config, $config_proto, $executor_type),
            (CallOp $f, $args),
          [(ArgTypesMatchCallee $op, $args, $f)]>;
}

// The extra attr on this op is _disable_call_shape_inference, which we ignore
// in the bridge.
def : Pat<(TF_LegacyCallOp:$op $args, FlatSymbolRefAttr:$f, $attr),
          (CallOp $f, $args),
        [(ArgTypesMatchCallee $op, $args, $f)]>;

//===----------------------------------------------------------------------===//
// Reverse op patterns.
//===----------------------------------------------------------------------===//

// Handles axis conversion for TF reverse.
def ConvertAxisAttr : NativeCodeCall<"ConvertAxisAttr($0, $1.cast<ElementsAttr>(), &$_builder)">;

def : Pat<(TF_ReverseV2Op AnyRankedTensor:$values, (ConstantLikeMatcher ElementsAttr:$axis)),
    (HLO_ReverseOp $values, (ConvertAxisAttr $values, $axis))>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

foreach Mapping = [
                   [TF_AbsOp, HLO_AbsOp],
                   [TF_AcosOp, CHLO_AcosOp],
                   [TF_AcoshOp, CHLO_AcoshOp],
                   [TF_AsinOp, CHLO_AsinOp],
                   [TF_AsinhOp, CHLO_AsinhOp],
                   [TF_AtanOp, CHLO_AtanOp],
                   [TF_AtanhOp, CHLO_AtanhOp],
                   [TF_CeilOp, HLO_CeilOp],
                   [TF_CoshOp, CHLO_CoshOp],
                   [TF_ComplexAbsOp, HLO_AbsOp],
                   [TF_ConjOp, CHLO_ConjOp],
                   [TF_CosOp, HLO_CosOp],
                   [TF_DigammaOp, CHLO_DigammaOp],
                   [TF_ExpOp, HLO_ExpOp],
                   [TF_Expm1Op, HLO_Expm1Op],
                   [TF_ErfOp, CHLO_ErfOp],
                   [TF_ErfcOp, CHLO_ErfcOp],
                   [TF_FloorOp, HLO_FloorOp],
                   [TF_ImagOp, HLO_ImagOp],
                   [TF_InvertOp, HLO_NotOp],
                   [TF_IsFiniteOp, HLO_IsFiniteOp],
                   [TF_IsInfOp, CHLO_IsInfOp],
                   [TF_LgammaOp, CHLO_LgammaOp],
                   [TF_LogOp, HLO_LogOp],
                   [TF_Log1pOp, HLO_Log1pOp],
                   [TF_LogicalNotOp, HLO_NotOp],
                   [TF_NegOp, HLO_NegOp],
                   [TF_RealOp, HLO_RealOp],
                   [TF_RsqrtOp, HLO_RsqrtOp],
                   [TF_SigmoidOp, HLO_LogisticOp],
                   [TF_SinhOp, CHLO_SinhOp],
                   [TF_SinOp, HLO_SinOp],
                   [TF_SqrtOp, HLO_SqrtOp],
                   [TF_TanhOp, HLO_TanhOp],
                   [TF_TanOp, CHLO_TanOp]
                  ] in {
 def : Pat<(Mapping[0] HLO_Tensor:$input),
           (Mapping[1] $input)>;
}

def : Pat<(TF_AngleOp $x), (HLO_Atan2Op (HLO_ImagOp $x), (HLO_RealOp $x))>;

// TODO(bixia): Lower with Truncate=True for floating point value conversions.
def : Pat<(TF_CastOp $arg, ConstBoolAttrFalse), (HLO_ConvertOp $arg)>;

def : Pat<(TF_TransposeOp:$res $arg, (ConstantLikeMatcher ElementsAttr:$permutation)),
          (HLO_TransposeOp $arg, (CastElementsToI64Elements $permutation))>;


// Lowering these ops with static shape to mhlo.reshape
foreach TfOp = [TF_ExpandDimsOp, TF_ReshapeOp, TF_SqueezeOp, ] in {
  def : Pat<(TfOp:$res HLO_Tensor:$arg, $ignored),
            (HLO_ReshapeOp $arg), [(AnyStaticShapeTensor $res)],
            (addBenefit 2)>;
}

// Lowering tf.Reshape with dynamic shape
def : Pat<(TF_ReshapeOp:$res HLO_Tensor:$arg, $shape),
          (CHLO_DynamicReshapeOp $arg, $shape)>;

// Returns NaN if x is NaN, 0 if x is 0, -1 if x < 0 and 1 if x > 0.
def : Pat<(TF_SignOp $x), (HLO_SignOp $x)>;

def BothElementTypesSameWidthIntOrFloat : Constraint<CPred<
  "getElementTypeOrSelf($0.getType()).isIntOrFloat() && "
  "getElementTypeOrSelf($1.getType()).isIntOrFloat()">,
  "element types must be integers or floats">;

// TODO(mgester): Due to restrictions of xla::BitcastConvertType we currently
// only lower if both input and output types are int or float and have same width

def : Pat<(TF_BitcastOp:$res HLO_Tensor:$arg),
          (HLO_BitcastConvertOp $arg),
          [(BothElementTypesSameWidthIntOrFloat $res, $arg)]>;

// TODO(jpienaar): Lower constant like to constant to broadcast if dynamic
// and going to MHLO.

//===----------------------------------------------------------------------===//
// Random ops.
//===----------------------------------------------------------------------===//

foreach srcDstOpPair = [[TF_RandomUniformOp, HLO_RngUniformOp],
                        [TF_RandomStandardNormalOp, HLO_RngNormalOp]] in {
// TODO(b/148269299): handle random number generator seeds/states correctly.
def : Pat<(srcDstOpPair[0]:$old $shape, $seed, $seed2),
          (srcDstOpPair[1]
            (HLO_ConstantOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.dtype(), 0.0)">)),
            (HLO_ConstantOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.dtype(), 1.0)">)),
            (CastValueToI64 $old, $shape)),
          [(IsShapedTensor $shape)]>;
}

//===----------------------------------------------------------------------===//
// Sigmoid grad op.
//===----------------------------------------------------------------------===//

// TODO(hinsu): Handle unranked inputs by broadcasting constant one to the
// shape of $l instead of having it as a constant.
def : Pat<(TF_SigmoidGradOp AnyRankedTensor:$l, AnyRankedTensor:$r),
          (HLO_MulOp
           (HLO_MulOp $r, $l),
           (HLO_SubOp (HLO_ConstantOp (ConstantSplat<"1"> $l)), $l))>;

//===----------------------------------------------------------------------===//
// Softplus op.
//===----------------------------------------------------------------------===//

def EpsilonValue : NativeCodeCall<"GetEpsilonValue($0.getType())">;

def : Pattern<(TF_SoftplusOp AnyTensor:$features),
              [
                (HLO_ExpOp:$features_exp $features),
                (CHLO_BroadcastAddOp:$threshold
                 (HLO_LogOp (HLO_ConstantOp (EpsilonValue $features))),
                 (HLO_ConstantOp (GetScalarOfType<2> $features)),
                 (NullDenseIntElementsAttr)
                ),
                (HLO_SelectOp:$output
                 (CHLO_BroadcastCompareOp
                  $features,
                  (HLO_NegOp $threshold),
                  (NullDenseIntElementsAttr),
                  HLO_ComparisonDirectionValue<"GT">,
                  (HLO_DEFAULT_COMPARISON_TYPE)
                 ),
                 $features,
                 (HLO_SelectOp
                  (CHLO_BroadcastCompareOp
                   $features,
                   $threshold,
                   (NullDenseIntElementsAttr),
                   HLO_ComparisonDirectionValue<"LT">,
                   (HLO_DEFAULT_COMPARISON_TYPE)
                  ),
                  $features_exp,
                  (HLO_Log1pOp $features_exp)
                 )
                ),
                (replaceWithValue $output)
              ]>;

//===----------------------------------------------------------------------===//
// XlaReplicaId op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaReplicaIdOp),
          (TF_CastOp (HLO_ReplicaIdOp), /*truncate=*/ConstBoolAttrFalse)>;

//===----------------------------------------------------------------------===//
// XlaGather op.
//===----------------------------------------------------------------------===//

def ToGatherDimNumsAttr : NativeCodeCall<"GetGatherDimNumsAttr($0, &$_builder)">;

def HasValidGatherDims : Constraint<CPred<"HasValidGatherDims($0)">>;

def : Pat<(TF_XlaGatherOp $operand, $start_indices, (ConstantLikeMatcher ElementsAttr:$slice_sizes),
                          $dimension_numbers, $indices_are_sorted),
          (HLO_GatherOp $operand, $start_indices,
                        (ToGatherDimNumsAttr $dimension_numbers),
                        (CastElementsToI64Elements $slice_sizes),
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
          (HLO_DotGeneralOp $lhs, $rhs,
                            (ToDotDimNumsAttr $dimension_numbers),
                            (ToPrecisionConfigsAttr $precision_config)),
          [(HasValidDotDims $dimension_numbers), (HasValidPrecisionConfig $precision_config)]>;

//===----------------------------------------------------------------------===//
// XlaDotV2Op op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaDotV2Op $lhs, $rhs, $dimension_numbers, $precision_config),
          (HLO_DotGeneralOp $lhs, $rhs,
                            (ToDotDimNumsAttr $dimension_numbers),
                            (ToPrecisionConfigsAttr $precision_config)),
          [(HasValidDotDims $dimension_numbers), (HasValidPrecisionConfig $precision_config)]>;

//===----------------------------------------------------------------------===//
// XlaDynamicSlice op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaDynamicSliceOp:$op HLO_Tensor:$input, HLO_Tensor:$starting_indices,
           (ConstantLikeMatcher AnyAttr:$slice_sizes)),
          (HLO_DynamicSliceOp $input,
           (UnpackStartingIndices $op, $starting_indices),
           (TFSliceSizes2HLOSliceSizes $input, $starting_indices, $slice_sizes))>;

//===----------------------------------------------------------------------===//
// XlaEisumOp op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaEinsumOp $lhs, $rhs, $equation),
          (HLO_EinsumOp $lhs, $rhs, $equation)>;

//===----------------------------------------------------------------------===//
// XlaOptimizationBarrierOp op.
//===----------------------------------------------------------------------===//

def : Pat<(TF_XlaOptimizationBarrierOp $args),
          (HLO_OptimizationBarrierOp $args)>;
