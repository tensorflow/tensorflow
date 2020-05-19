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
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/xla/ir/chlo_ops.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops.td"

def SignedIntTensor : TensorOf<[I1, I8, I16, I32, I64]>;

// IEEE compliant floating point tensors.
def IEEEFloatTensor : TensorOf<[F16, F32, F64]>;

//===----------------------------------------------------------------------===//
// BatchNorm op patterns.
//===----------------------------------------------------------------------===//

def FeatureDimension : NativeCodeCall<
    "getFeatureDimensionAttr($_builder, $0, $1)">;
def FalseBoolAttr : AttrConstraint<CPred<"!$_self.getValue()">>;
def TrueBoolAttr : AttrConstraint<CPred<"$_self.getValue()">>;

def CastValueToI64: NativeCodeCall<
  "CastValueToI64($0.getLoc(), $1, &$_builder)">;

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

def : Pattern<
    (TF_FusedBatchNormOp:$root $x, $scale, $offset, $mean, $variance, $epsilon,
                         $exponential_avg_factor, $data_format,
                         FalseBoolAttr:$is_training),
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
  : Pat<(FromOp AnyRankedTensor:$l, AnyRankedTensor:$r),
        (ToOp $l, $r, (BinBroadcastDimensions $l, $r))>;

foreach fromToBinPair = [[TF_AddOp, HLOClient_BroadcastAddOp],
                         [TF_AddV2Op, HLOClient_BroadcastAddOp],
                         [TF_DivOp, HLOClient_BroadcastDivOp],
                         [TF_LeftShiftOp, HLOClient_BroadcastShiftLeftOp],
                         [TF_MaximumOp, HLOClient_BroadcastMaxOp],
                         [TF_MinimumOp, HLOClient_BroadcastMinOp],
                         [TF_MulOp, HLOClient_BroadcastMulOp],
                         [TF_PowOp, HLOClient_BroadcastPowOp],
                         [TF_RealDivOp, HLOClient_BroadcastDivOp],
                         [TF_SubOp, HLOClient_BroadcastSubOp]] in
  def : DirectBinaryPat<fromToBinPair[0], fromToBinPair[1]>;

def LowerRightShiftSigned :
  Pat<(TF_RightShiftOp AnyRankedTensor:$l, AnyRankedTensor:$r),
      (HLOClient_BroadcastShiftRightArithmeticOp $l, $r,
       (BinBroadcastDimensions $l, $r)),
      [(SignedIntTensor $r)]>;

// TODO(hinsu): Lower unsigned types to HLO_ShiftRightLogical once the HLO op
// supports unsigned integers.

def : Pat<(TF_ComplexOp $r, $i), (HLO_ComplexOp $r, $i)>;

// Performs a substitution of FloorDiv, pseudo code below:
//
//  return floor(div(x, y))
def : Pat<(TF_FloorDivOp AnyRankedTensor:$l, AnyRankedTensor:$r),
          (HLO_FloorOp
           (HLOClient_BroadcastDivOp $l, $r, (BinBroadcastDimensions $l, $r))),
          [(IEEEFloatTensor $l)]>;

// Performs a substitution of FloorDiv for integer tensors, which required
// additional correction for a negative numerator / denominator. Equivalent
// pseudocode is shown below:
//
// if ((x < 0) != (y < 0)) {
//   T abs_x = std::abs(x);
//   T abs_y = std::abs(y);
//   return -(abs_x + abs_y - 1) / abs_y;
// } else {
//   return x / y;
// }
//
// BraodcastToDimensions is used to compute the broadcast attr to higher
// dimensions. This computes the broadcast of 'l' to broadcast('l', 'r')
// without returning the broadcast of 'r' to broadcast('l', 'r').
//
// NOTE: This should be optimized for unsigned integers.
// Requires static shaped inputs to create constant splats and computation of
// broadcast attributes.
def : Pat<(TF_FloorDivOp AnyStaticShapeTensor:$l, AnyStaticShapeTensor:$r),
        (HLO_SelectOp
         (HLOClient_BroadcastCompareOp
          (HLOClient_BroadcastCompareOp $l, (HLO_ConstOp (ConstantSplat<"0"> $l)),
           (NullDenseIntElementsAttr), HLO_COMPARISON_DIRECTION_LT),
          (HLOClient_BroadcastCompareOp $r, (HLO_ConstOp (ConstantSplat<"0"> $r)),
           (NullDenseIntElementsAttr), HLO_COMPARISON_DIRECTION_LT),
          (BinBroadcastDimensions $l, $r), HLO_COMPARISON_DIRECTION_EQ),
        (HLOClient_BroadcastDivOp $l, $r, (BinBroadcastDimensions $l, $r)),
          (HLOClient_BroadcastDivOp
           (HLO_NegOp:$neg (HLOClient_BroadcastAddOp (HLO_AbsOp $l),
                       (HLOClient_BroadcastSubOp (HLO_AbsOp $r),
                        (HLO_ConstOp (ConstantSplat<"1"> $r)),
                        (NullDenseIntElementsAttr)),
                     (BinBroadcastDimensions $l, $r))),
           (HLO_AbsOp:$abs $r), (BinBroadcastDimensions $neg, $abs))),
        [(SignedIntTensor $l)]>;

// Performs a substitution of FloorMod designed to correct for possibly negative
// values. Pseudocode shown below:
//
//   T trunc_mod = std::fmod(x, y);
//   return trunc_mod != 0 && (y < 0 != trunc_mod < 0) ? trunc_mod + y
// Requires static shaped inputs to create constant splats and computation of
// broadcast attributes.
def : Pat<(TF_FloorModOp AnyStaticShapeTensor:$l, AnyStaticShapeTensor:$r),
      (HLO_SelectOp
       (HLOClient_BroadcastAndOp
        (HLOClient_BroadcastCompareOp
         (HLOClient_BroadcastRemOp:$rem $l, $r, (BinBroadcastDimensions $l, $r)),
         (HLO_ConstOp:$l_zeros (ConstantSplat<"0"> $l)),
         (BinBroadcastDimensions $l, $rem), HLO_COMPARISON_DIRECTION_NE),
        (HLOClient_BroadcastCompareOp
         (HLOClient_BroadcastCompareOp:$r_cmp $r,
          (HLO_ConstOp:$r_zeros (ConstantSplat<"0"> $r)),
          (NullDenseIntElementsAttr), HLO_COMPARISON_DIRECTION_LT),
         (HLOClient_BroadcastCompareOp:$rem_cmp $rem, $r_zeros,
          (BinBroadcastDimensions $rem, $r_zeros), HLO_COMPARISON_DIRECTION_LT),
         (BinBroadcastDimensions $r_cmp, $rem_cmp), HLO_COMPARISON_DIRECTION_NE),
        (NullDenseIntElementsAttr)),
        (HLOClient_BroadcastAddOp $r,
         $rem, (BinBroadcastDimensions $r, $rem)), $rem)>;

//===----------------------------------------------------------------------===//
// Logical & bitwise binary op patterns.
//===----------------------------------------------------------------------===//

class DirectLogicalBinaryPat<Op FromOp, Op ToOp>
  : Pat<(FromOp AnyRankedTensor:$l, AnyRankedTensor:$r),
        (ToOp $l, $r, (BinBroadcastDimensions $l, $r)),
        [(SignedIntTensor $l)]>;

foreach fromToBinPair = [[TF_LogicalAndOp, HLOClient_BroadcastAndOp],
                         [TF_LogicalOrOp, HLOClient_BroadcastOrOp],
                         [TF_BitwiseOrOp, HLOClient_BroadcastOrOp],
                         [TF_BitwiseAndOp, HLOClient_BroadcastAndOp]] in
  def : DirectLogicalBinaryPat<fromToBinPair[0], fromToBinPair[1]>;

//===----------------------------------------------------------------------===//
// Compare op patterns.
//===----------------------------------------------------------------------===//

class DirectComparePat<Op FromOp, StrEnumAttrCase direction>
  : Pat<(FromOp AnyRankedTensor:$l, AnyRankedTensor:$r),
        (HLOClient_BroadcastCompareOp
           $l, $r, (BinBroadcastDimensions $l, $r), direction)>;

def : DirectComparePat<TF_GreaterOp, HLO_COMPARISON_DIRECTION_GT>;
def : DirectComparePat<TF_GreaterEqualOp, HLO_COMPARISON_DIRECTION_GE>;
def : DirectComparePat<TF_LessOp, HLO_COMPARISON_DIRECTION_LT>;
def : DirectComparePat<TF_LessEqualOp, HLO_COMPARISON_DIRECTION_LE>;

class EqualityPat<Op FromOp, StrEnumAttrCase direction>
    : Pat<(FromOp AnyRankedTensor:$l, AnyRankedTensor:$r,
           TrueBoolAttr:$incompatible_shape_error),
        (HLOClient_BroadcastCompareOp
         $l, $r, (BinBroadcastDimensions $l, $r), direction),
        [(AreBroadcastCompatible $l, $r)]>;

def : EqualityPat<TF_EqualOp, HLO_COMPARISON_DIRECTION_EQ>;
def : EqualityPat<TF_NotEqualOp, HLO_COMPARISON_DIRECTION_NE>;

//===----------------------------------------------------------------------===//
// Complex op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_ConjOp $v),
          (HLO_ComplexOp (HLO_RealOp $v), (HLO_NegOp (HLO_ImagOp $v)))>;

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
def : Pat<(TF_ConcatV2Op $inputs, (TF_ConstOp OneElementAttr:$axis)),
          (HLO_ConcatenateOp $inputs,
            (GetHLOAxisFromTFAxisVariadic $axis, $inputs)),
          [(HasRankedFirstOperand $inputs)]>;

//===----------------------------------------------------------------------===//
// CrossReplicaSum op patterns.
//===----------------------------------------------------------------------===//

def CastElementsToI64Elements : NativeCodeCall<
  "xla::ConvertElementsAttr("
    "$0, $_builder.getIntegerType(64)).cast<DenseIntElementsAttr>()">;

def : Pat<(TF_CrossReplicaSumOp $input, (TF_ConstOp $group_assignment)),
          (HLO_CrossReplicaSumOp $input,
            (CastElementsToI64Elements $group_assignment))>;

//===----------------------------------------------------------------------===//
// All2All op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_AllToAllOp AnyRankedTensor:$input, (TF_ConstOp $group_assignment), I64Attr:$concat_dimension, $split_dimension, $split_count),
          (HLO_AllToAllOp $input, $split_dimension, $concat_dimension, $split_count, (CastElementsToI64Elements $group_assignment))>;

//===----------------------------------------------------------------------===//
// FFT op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_RFFTOp $input, (TF_ConstOp I32ElementsAttr:$fft_length)),
          (HLO_FftOp $input, HLO_FFT_TYPE_RFFT,
           (CastElementsToI64Elements $fft_length))>;

//===----------------------------------------------------------------------===//
// GatherV2 op patterns.
//===----------------------------------------------------------------------===//

// Here, $params and $indices needs to be ranked so that $axis and $batch_dims
// attributes can be converted from TensorFlow axis format supporting negative
// indexing to the HLO format.
def LegalizeGatherV2 :
  Pat<(TF_GatherV2Op AnyRankedTensor:$params, AnyRankedTensor:$indices,
        (TF_ConstOp $axis), $batch_dims),
      (HLO_TorchIndexSelectOp $params, $indices,
        (GetHLOAxisFromTFAxis $axis, $params),
        (GetHLOAxisFromTFAxis $batch_dims, $indices))>;

//===----------------------------------------------------------------------===//
// Pad op patterns.
//===----------------------------------------------------------------------===//

class SliceDenseIntElementsAttrColumn2D<string column> : NativeCodeCall<
  "SliceDenseIntElementsAttrColumn2D($0, " # column # " )">;

class SliceDenseIntElementsAttr<string index, string axis> : NativeCodeCall<
  "SliceDenseIntElementsAttr($0, " # index # ", " # axis # ")">;

// Interior padding attribute based on the TF padding.
def GetInteriorPadding : NativeCodeCall <
  "GetInteriorPadding($0)">;

def : Pat<(TF_PadV2Op $input, (TF_ConstOp $padding), $c),
          (HLO_PadOp $input, $c,
           (SliceDenseIntElementsAttrColumn2D<"0"> $padding),
           (SliceDenseIntElementsAttrColumn2D<"1"> $padding),
           (GetInteriorPadding $padding))>;

//===----------------------------------------------------------------------===//
// Identity op patterns.
//===----------------------------------------------------------------------===//

foreach src = [TF_IdentityOp, TF_StopGradientOp] in
  def : Pat<(src $op), (replaceWithValue $op)>;
def : Pat<(TF_PreventGradientOp $op, $msg), (replaceWithValue $op)>;

//===----------------------------------------------------------------------===//
// MatMul op patterns.
//===----------------------------------------------------------------------===//

def Get2DTransposePerm: NativeCodeCall<
  "Get2DTransposePerm($0, &$_builder)">;

def : Pat<(TF_MatMulOp $a, $b, $transpose_a, $transpose_b),
          (HLO_DotOp
          (TF_TransposeOp $a, (TF_ConstOp (Get2DTransposePerm $transpose_a))),
          (TF_TransposeOp $b, (TF_ConstOp (Get2DTransposePerm $transpose_b))),
          /*precision_config=*/(NullArrayAttr))>;


//===----------------------------------------------------------------------===//
// SparseMatMul op patterns.
//===----------------------------------------------------------------------===//

// Ignores the sparseness hints and translates tf.SparseMatMul to tf.MatMul
// until we will have an implementation that can use the information.
def SparseMatMulToMatMul : Pat<(TF_SparseMatMulOp $a, $b, $a_sparse, $b_sparse,
                                $transpose_a, $transpose_b),
                               (TF_MatMulOp $a, $b, $transpose_a,
                                $transpose_b)>;

//===----------------------------------------------------------------------===//
// MatrixBandPart op pattern.
//===----------------------------------------------------------------------===//

class getIntegerAttr<string x>: NativeCodeCall<
  "$_builder.getI64IntegerAttr(" # x # ")">;

class GetDimensionSizeFromEnd<string dimFromEnd>: NativeCodeCall<
  "$_builder.getI64IntegerAttr(GetDimensionSizeFromEnd($0, " # dimFromEnd # "))"
  >;

// TODO(b/149615308): Enable IotaOp usage as a child operation in a pattern
// For now, this op needs to be created in C++ because the expected output type
// cannot be inferred.
class createIotaOp<string dim>: NativeCodeCall<
  "$_builder.create<xla_hlo::IotaOp>($0.getOwner()->getLoc(), "
  "Get2DTensorType($1), $_builder.getI64IntegerAttr(" # dim # "))">;

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
def : Pattern<(TF_MatrixBandPartOp:$op AnyRankedTensor:$input, $num_lower, $num_upper),
         [(HLO_ConstOp:$m_dim (GetDimensionSizeFromEnd<"0"> $input)),
          (HLO_ConstOp:$n_dim (GetDimensionSizeFromEnd<"1"> $input)),
          (HLO_SelectOp:$num_lower_or_m
           (HLO_CompareOp
            $num_lower, (HLO_ConstOp:$zero (ConstantSplat<"0"> $num_lower)),
            HLO_COMPARISON_DIRECTION_LT
           ),
           $m_dim,
           $num_lower
          ),
          (HLO_SelectOp:$num_upper_or_n
           (HLO_CompareOp
            $num_upper, $zero, HLO_COMPARISON_DIRECTION_LT
           ),
           $n_dim,
           $num_upper
          ),
          (HLO_SelectOp
           (HLO_AndOp
            (HLOClient_BroadcastCompareOp
             (HLO_NegOp
              (createConvertOp $op, $num_lower_or_m, $input)
             ),
             (HLO_SubOp:$offset
              (createIotaOp<"1"> $op, $input), (createIotaOp<"0"> $op, $input)
             ),
             (NullDenseIntElementsAttr), HLO_COMPARISON_DIRECTION_LE
            ),
            (HLOClient_BroadcastCompareOp
             $offset,
             (createConvertOp
              $op, $num_upper_or_n, $input
             ),
             (NullDenseIntElementsAttr), HLO_COMPARISON_DIRECTION_LE
            )
           ),
           $input,
           (HLO_ConstOp (ConstantSplat<"0"> $input))
          )]>;

//===----------------------------------------------------------------------===//
// Nullary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_ConstOp:$res ElementsAttr:$value),
          (TensorCastOp (HLO_ConstOp $value)),
          [(HLO_Tensor $res)]>;

//===----------------------------------------------------------------------===//
// Relu op patterns.
//===----------------------------------------------------------------------===//

// TODO(hinsu): Make these patterns to TF to TF lowering. Relu6 lowering will
// require HLO canonicalization of min and max on a tensor to ClampOp.

// TODO(hinsu): Lower unsigned and quantized types after supporting
// them in GetScalarOfType.
def : Pat<(TF_ReluOp AnyRankedTensor:$input),
          (HLOClient_BroadcastMaxOp
               (HLO_ConstOp:$zero (GetScalarOfType<0> $input)), $input,
               (BinBroadcastDimensions $zero, $input)),
          [(TF_SintOrFpTensor $input)]>;

// TODO(hinsu): Lower unsigned and quantized types after supporting
// them in GetScalarOfType.
def : Pat<(TF_Relu6Op AnyRankedTensor:$input),
          (HLO_ClampOp (HLO_ConstOp (GetScalarOfType<0> $input)), $input,
                       (HLO_ConstOp (GetScalarOfType<6> $input))),
          [(TF_SintOrFpTensor $input)]>;

// ReluGrad(gradients, features) = gradients * (features > 0)
//
// $gradients needs to be of static shape so that on_true and on_false operands
// of SelectOp have same shape.
//
// $features needs to be ranked for computation of the broadcast dimensions for
// CompareOp.
//
// TODO(hinsu): Relax $gradients static shape requirement when there is a way
// to create splat tensor of dynamic shape in HLO.
def : Pat<(TF_ReluGradOp AnyStaticShapeTensor:$gradients, AnyRankedTensor:$features),
          (HLO_SelectOp
            (HLOClient_BroadcastCompareOp $features,
              (HLO_ConstOp (GetScalarOfType<0> $features)),
              (NullDenseIntElementsAttr), HLO_COMPARISON_DIRECTION_GT),
            $gradients, (HLO_ConstOp (ConstantSplat<"0"> $gradients)))>;

//===----------------------------------------------------------------------===//
// Slice op patterns.
//===----------------------------------------------------------------------===//

def CastToI64AndUnpackTensor: NativeCodeCall<
  "UnpackTensorAlongZeroDim($0.getLoc(), CastValueToI64($0.getLoc(), $1, &$_builder), &$_builder).output()">;

def CanBeTranslatedToDynamicSlice : Constraint<CPred<
  "CanBeTranslatedToDynamicSlice($0, $1, $2.cast<DenseIntElementsAttr>())">>;

def TFSliceSizes2HLOSliceSizes : NativeCodeCall<
    "TFSliceSizes2HLOSliceSizes($0, $1, $2.cast<DenseIntElementsAttr>(),"
    "&$_builder)">;

def : Pat<(TF_SliceOp:$op HLO_Tensor:$input, HLO_Tensor:$starting_indices,
           (TF_ConstOp $slice_sizes)),
          (HLO_DynamicSliceOp $input,
           (CastToI64AndUnpackTensor $op, $starting_indices),
           (TFSliceSizes2HLOSliceSizes $input, $starting_indices, $slice_sizes)),
          [(CanBeTranslatedToDynamicSlice $input, $starting_indices,
            $slice_sizes)]>;

//===----------------------------------------------------------------------===//
// PartitionedCall op patterns.
//===----------------------------------------------------------------------===//

def ArgTypesMatchCallee : Constraint<
    CPred<"ArgTypesMatchCallee($0[0].getOwner(), $1, $2)">>;

foreach callOp = [TF_PartitionedCallOp, TF_StatefulPartitionedCallOp] in {
  def : Pat<(callOp:$op $args, FlatSymbolRefAttr:$f,
             $config, $config_proto, $executor_type),
            (CallOp $f, $args),
          [(ArgTypesMatchCallee $op, $args, $f)]>;
}

//===----------------------------------------------------------------------===//
// Reverse op patterns.
//===----------------------------------------------------------------------===//

// Handles axis conversion for TF reverse.
def ConvertAxisAttr : NativeCodeCall<"ConvertAxisAttr($0, $1, &$_builder)">;

def : Pat<(TF_ReverseV2Op AnyRankedTensor:$values, (TF_ConstOp $axis)),
    (HLO_ReverseOp $values, (ConvertAxisAttr $values, $axis))>;

//===----------------------------------------------------------------------===//
// Ternary op patterns.
//===----------------------------------------------------------------------===//

def BothTypesMatch : Constraint<CPred<"$0.getType() == $1.getType()">,
   "types must be equal">;

def : Pat<(TF_SelectOp $cond, $t, $e), (HLO_SelectOp $cond, $t, $e),
  // TODO(jpienaar): This restriction is to avoid creating a currently
  // unsupported HLO select.
  [(BothTypesMatch $t, $e)]>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

foreach Mapping = [
                   [TF_AbsOp, HLO_AbsOp],
                   [TF_CeilOp, HLO_CeilOp],
                   [TF_ComplexAbsOp, HLO_AbsOp],
                   [TF_CosOp, HLO_CosOp],
                   [TF_ExpOp, HLO_ExpOp],
                   [TF_FloorOp, HLO_FloorOp],
                   [TF_ImagOp, HLO_ImagOp],
                   [TF_IsFiniteOp, HLO_IsFiniteOp],
                   [TF_LogOp, HLO_LogOp],
                   [TF_Log1pOp, HLO_Log1pOp],
                   [TF_LogicalNotOp, HLO_NotOp],
                   [TF_NegOp, HLO_NegOp],
                   [TF_RealOp, HLO_RealOp],
                   [TF_RsqrtOp, HLO_RsqrtOp],
                   [TF_SinOp, HLO_SinOp],
                   [TF_SqrtOp, HLO_SqrtOp],
                   [TF_TanhOp, HLO_TanhOp],
                  ] in {
 def : Pat<(Mapping[0] HLO_Tensor:$input),
           (Mapping[1] $input)>;
}

// TODO(bixia): Lower Cast with a Complex type source operand or with
// Truncate=True for floating point value conversions.
def : Pat<(TF_CastOp HLO_Tensor:$arg, ConstBoolAttrFalse),
          (HLO_ConvertOp $arg)>;

def : Pat<(TF_TransposeOp:$res $arg, (TF_ConstOp $permutation)),
          (HLO_TransposeOp $arg, (CastElementsToI64Elements $permutation))>;

// Result of the following ops changing tensor shape needs to have static
// shape as HLO doesn't yet support dynamic reshaping ops.
//
// TODO(hinsu): Update once HLO supports dynamic reshaping ops.
foreach TfOp = [TF_ExpandDimsOp, TF_ReshapeOp, TF_SqueezeOp, ] in {
  def : Pat<(TfOp:$res $arg, $ignored),
            (HLO_ReshapeOp $arg), [(AnyStaticShapeTensor $res)]>;
}

// Returns 0 if x is NaN, 0 if x is 0, -1 if x < 0 and 1 if x > 0.
def : Pat<(TF_SignOp $x),
          (HLO_SelectOp
            (HLO_CompareOp
              $x,
              $x,
              HLO_COMPARISON_DIRECTION_NE
            ),
            (HLO_ConstOp (ConstantSplat<"0"> $x)),
            (HLO_SignOp $x)
          )>;

def BothElementTypesSameWidthIntOrFloat : Constraint<CPred<
  "getElementTypeOrSelf($0.getType()).isSignlessIntOrFloat() && "
  "getElementTypeOrSelf($1.getType()).isSignlessIntOrFloat() && "
  "getElementTypeOrSelf($0.getType()).getIntOrFloatBitWidth() == "
  "getElementTypeOrSelf($1.getType()).getIntOrFloatBitWidth()">,
  "element types must be integers or floats of same width">;

// TODO(mgester): Due to restrictions of xla::BitcastConvertType we currently
// only lower if both input and output types are int or float and have same width

def : Pat<(TF_BitcastOp:$res HLO_Tensor:$arg),
          (HLO_BitcastConvertOp $arg),
          [(BothElementTypesSameWidthIntOrFloat $res, $arg)]>;

//===----------------------------------------------------------------------===//
// Random ops.
//===----------------------------------------------------------------------===//

foreach srcDstOpPair = [[TF_RandomUniformOp, HLO_RngUniformOp],
                        [TF_RandomStandardNormalOp, HLO_RngNormalOp]] in {
// TODO(b/148269299): handle random number generator seeds/states correctly.
def : Pat<(srcDstOpPair[0]:$old $shape, $seed, $seed2),
          (srcDstOpPair[1]
            (HLO_ConstOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.dtype(), 0.0)">)),
            (HLO_ConstOp
              (NativeCodeCall<"$_builder.getFloatAttr(old.dtype(), 1.0)">)),
            (CastValueToI64 $old, $shape)),
          [(IsShapedTensor $shape)]>;
}

//===----------------------------------------------------------------------===//
// Sigmoid grad op.
//===----------------------------------------------------------------------===//
def : Pat<(TF_SigmoidGradOp AnyRankedTensor:$l, AnyRankedTensor:$r),
          (HLO_MulOp
           (HLO_MulOp $r, $l),
           (HLO_SubOp (HLO_ConstOp (ConstantSplat<"1"> $l)), $l)),
          [(IEEEFloatTensor $l)]>;
