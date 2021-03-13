/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This is the legalization pattern definition file for HLO to TF.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.td"
include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"

// Check if broadcasting is compatible with TF ops.
def IsLegalNumpyRankedBroadcast :
    Constraint<CPred<"hlo::IsLegalNumpyRankedBroadcast($0, $1, $2)">,
    "broadcasting should be compatible with TF ops">;

// Return a constant op that carries the shape of the given value.
def ShapeToConst : NativeCodeCall<"ShapeToConst($_builder, $0)">;

// Check if broadcast dimensions match Tensorflow convention.
def IsTFStyleBroadcast : Constraint<CPred<"IsTFStyleBroadcast($0, $1)">,
    "new dimensions are added as prefix">;

// Check if broadcast dimensions do not match Tensorflow convention.
def IsNotTFStyleBroadcast : Constraint<Neg<CPred<"IsTFStyleBroadcast($0, $1)">>,
    "new dimensions are inserted in intermediate positions">;

// Return intermediate shape before broadcasting, wrapped in a constant op.
def ExpandedShape : NativeCodeCall<"ExpandedShape($_builder, $0, $1, $2)">;

def : Pat<(HLO_ConstOp $value), (TF_ConstOp $value)>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
// Note that these are legalized from chlo.broadcast_* ops, since those are
// semantically compatible with the corresponding TF ops. Depending on
// context, getting to these ops may require some raising.
//===----------------------------------------------------------------------===//

foreach fromToBinPair = [[HLO_AddOp, HLOClient_BroadcastAddOp, TF_AddV2Op],
                         [HLO_DivOp, HLOClient_BroadcastDivOp, TF_DivOp],
                         [HLO_ShiftLeftOp, HLOClient_BroadcastShiftLeftOp, TF_LeftShiftOp],
                         [HLO_MaxOp, HLOClient_BroadcastMaxOp, TF_MaximumOp],
                         [HLO_MinOp, HLOClient_BroadcastMinOp, TF_MinimumOp],
                         [HLO_MulOp, HLOClient_BroadcastMulOp, TF_MulOp],
                         [HLO_PowOp, HLOClient_BroadcastPowOp, TF_PowOp],
                         [HLO_SubOp, HLOClient_BroadcastSubOp, TF_SubOp],
                         [HLO_Atan2Op, HLOClient_BroadcastAtan2Op, TF_Atan2Op],
                         [HLO_RemOp, HLOClient_BroadcastRemOp, TF_ModOp]] in {
  def : Pat<(fromToBinPair[0] $l, $r), (fromToBinPair[2] $l, $r)>;
  def : Pat<(fromToBinPair[1] $l, $r, $broadcast_dimensions),
            (fromToBinPair[2] $l, $r),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
}

foreach pair  = [[HLO_AndOp, HLOClient_BroadcastAndOp, TF_BitwiseAndOp],
                 [HLO_OrOp, HLOClient_BroadcastOrOp, TF_BitwiseOrOp],
                 [HLO_XorOp, HLOClient_BroadcastXorOp, TF_BitwiseXorOp]] in {
  def : Pat<(pair[0] TF_IntTensor:$l, TF_IntTensor:$r), (pair[2] $l, $r)>;
  def : Pat<(pair[1] TF_IntTensor:$l, TF_IntTensor:$r, $broadcast_dimensions),
            (pair[2] $l, $r),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
}

foreach pair  = [[HLO_AndOp, HLOClient_BroadcastAndOp, TF_LogicalAndOp],
                 [HLO_OrOp, HLOClient_BroadcastOrOp, TF_LogicalOrOp]] in {
  def : Pat<(pair[0] I1Tensor:$l, I1Tensor:$r), (pair[2] $l, $r)>;
  def : Pat<(pair[1] I1Tensor:$l, I1Tensor:$r, $broadcast_dimensions),
            (pair[2] $l, $r),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
}

def : Pat<(HLO_ShiftRightArithmeticOp $l, $r), (TF_RightShiftOp $l, $r)>;
def : Pat<(HLOClient_BroadcastShiftRightArithmeticOp $l, $r,
                                                     $broadcast_dimensions),
          (TF_RightShiftOp $l, $r),
          [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
def : Pat<(HLO_ShiftRightLogicalOp $l, $r), (TF_RightShiftOp $l, $r)>;
def : Pat<(HLOClient_BroadcastShiftRightLogicalOp $l, $r,
                                                  $broadcast_dimensions),
          (TF_RightShiftOp $l, $r),
          [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;

def : Pat<(HLO_FloorOp (HLO_DivOp $l, $r)), (TF_FloorDivOp $l, $r)>;
def : Pat<(HLO_FloorOp (HLOClient_BroadcastDivOp $l, $r,
                                                 $broadcast_dimensions)),
          (TF_FloorDivOp $l, $r),
          [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;

def : Pat<(HLO_ComplexOp $r, $i), (TF_ComplexOp $r, $i)>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(HLO_ConvertOp HLO_Tensor:$operand),
          (TF_CastOp $operand, ConstBoolAttrFalse)>;

foreach Mapping = [[HLO_AbsOp, TF_AbsOp],
                   [HLO_BitcastConvertOp, TF_BitcastOp],
                   [HLO_CeilOp, TF_CeilOp],
                   [HLO_CosOp, TF_CosOp],
                   [HLO_ExpOp, TF_ExpOp],
                   [HLO_Expm1Op, TF_Expm1Op],
                   [HLO_FloorOp, TF_FloorOp],
                   [HLO_ImagOp, TF_ImagOp],
                   [HLO_IsFiniteOp, TF_IsFiniteOp],
                   [HLO_LogOp, TF_LogOp],
                   [HLO_Log1pOp, TF_Log1pOp],
                   [HLO_LogisticOp, TF_SigmoidOp],
                   [HLO_NotOp, TF_LogicalNotOp],
                   [HLO_NegOp, TF_NegOp],
                   [HLO_RealOp, TF_RealOp],
                   [HLO_RsqrtOp, TF_RsqrtOp],
                   [HLO_SinOp, TF_SinOp],
                   [HLO_SignOp, TF_SignOp],
                   [HLO_SqrtOp, TF_SqrtOp],
                   [HLO_TanhOp, TF_TanhOp]] in
 def : Pat<(Mapping[0] TF_IntOrFpTensor:$input), (Mapping[1] $input)>;

def : Pat<(HLO_AbsOp TF_ComplexTensor:$arg), (TF_ComplexAbsOp $arg)>;

def : Pat<(HLO_BroadcastOp $arg, $shape),
          (TF_BroadcastToOp $arg, (TF_ConstOp $shape))>;
def : Pat<(HLO_BroadcastInDimOp:$output $input, $broadcast_dimensions),
          (TF_BroadcastToOp $input, (ShapeToConst $output)),
          [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
def : Pat<(HLO_BroadcastInDimOp:$output $input, $broadcast_dimensions),
          (TF_BroadcastToOp
              (TF_ReshapeOp
                  $input,
                  (ExpandedShape $input, $broadcast_dimensions, $output)),
              (ShapeToConst $output)),
          [(IsNotTFStyleBroadcast $broadcast_dimensions, $output)]>;
def : Pat<(HLO_TransposeOp $arg, $permutation),
          (TF_TransposeOp $arg, (TF_ConstOp $permutation))>;
def : Pat<(HLO_ReverseOp $op, $dims), (TF_ReverseV2Op $op, (TF_ConstOp $dims))>;
def : Pat<(HLO_ReshapeOp:$output $input),
          (TF_ReshapeOp $input, (ShapeToConst $output))>;

//===----------------------------------------------------------------------===//
// Ternary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(HLO_ClampOp $min, $arg, $max),
          (TF_MaximumOp (TF_MinimumOp $arg, $max), $min)>;
def : Pat<(HLO_SelectOp $cond, $t, $e), (TF_SelectOp $cond, $t, $e)>;

//===----------------------------------------------------------------------===//
// Variadic op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(HLO_ConcatenateOp $inputs, $dim),
          (TF_ConcatV2Op $inputs, (TF_ConstOp $dim))>;

class HasCompareType<string value> :
    CPred<"$_self.cast<StringAttr>().getValue() == \"" # value # "\"">;

// Attribute value should be such that it matches the comparison used by
// TensorFlow, if the attribute is present.
def IsTFCompareType : AttrConstraint<
   Or<[CPred<"!$_self">, HasCompareType<"FLOAT">, HasCompareType<"SIGNED">,
       HasCompareType<"UNSIGNED">]>,
   "compare type supported by TensorFlow">;

//===----------------------------------------------------------------------===//
// Compare op patterns.
// Note that these are legalized from chlo.broadcast_* ops, since those are
// semantically compatible with the corresponding TF ops. Depending on
// context, getting to these ops may require some raising.
//===----------------------------------------------------------------------===//

foreach p = [[TF_EqualOp, HLO_COMPARISON_DIRECTION_EQ],
             [TF_NotEqualOp, HLO_COMPARISON_DIRECTION_NE]] in {
  def : Pat<(HLOClient_BroadcastCompareOp $l, $r, $broadcast_dimensions, p[1],
                                          IsTFCompareType:$type),
            (p[0] $l, $r, ConstBoolAttrTrue),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
  def : Pat<(HLO_CompareOp $l, $r, p[1], IsTFCompareType:$type),
            (p[0] $l, $r, ConstBoolAttrTrue)>;
}

foreach pair = [[TF_GreaterEqualOp, HLO_COMPARISON_DIRECTION_GE],
                [TF_GreaterOp, HLO_COMPARISON_DIRECTION_GT],
                [TF_LessEqualOp, HLO_COMPARISON_DIRECTION_LE],
                [TF_LessOp, HLO_COMPARISON_DIRECTION_LT]] in {
  def : Pat<(HLOClient_BroadcastCompareOp $l, $r, $broadcast_dimensions,
                                          pair[1], IsTFCompareType:$type),
            (pair[0] $l, $r),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
  def : Pat<(HLO_CompareOp $l, $r, pair[1], IsTFCompareType:$type),
            (pair[0] $l, $r)>;
}

def ConvertDotOp : NativeCodeCall<"ConvertDotOp($_builder, "
                                               "$0.getDefiningOp())">;
def : Pat<(HLO_DotOp:$old_value AnyStaticShapeTensor:$lhs,
               AnyStaticShapeTensor:$rhs, $precision_config),
          (ConvertDotOp $old_value)>;

def ConvertDotGeneralOp : NativeCodeCall<"ConvertDotGeneralOp($_builder, "
                                               "$0.getDefiningOp())">;
def : Pat<(HLO_DotGeneralOp:$old_value AnyStaticShapeTensor:$lhs,
               AnyStaticShapeTensor:$rhs, $dot_dimension_numbers,
               $precision_config),
          (ConvertDotGeneralOp $old_value)>;

def IsZero : Constraint<CPred<
  "$0.isSplat() && $0.getSplatValue<APInt>() == 0">>;
def ConvertPadOp : NativeCodeCall<
  "ConvertPadOp($_builder, $0.getDefiningOp())">;
def : Pat<(HLO_PadOp:$old_value $input, $pad_value, $pad_low, $pad_high,
               $pad_interior),
          (ConvertPadOp $old_value),
          [(IsZero $pad_interior)]>;

class FloatValueEquals<string val> : Constraint<CPred<
  "$0.isa<SplatElementsAttr>() && "
  "$0.cast<SplatElementsAttr>().getSplatValue<APFloat>().isExactlyValue(" # val # ")">>;
def SameValue : Constraint<CPred<"$0 == $1">>;
def FloatOrDefaultCompare : Constraint<CPred<
  "!$0 || $0.getValue() == \"FLOAT\"">>;

// Converts a soup of HLOs representing banker rounding (round x.5 to nearest
// even) to tf.round.
// The pattern matched executes the following computation:
// frac = x - floor(x)
// to_even = (floor(x) - 2 * floor(0.5 * x)) == 1
// if frac > 0.5 || (frac == 0.5 && to_even)
//   return floor + 1
// else
//   return floor
def : Pat<(HLO_SelectOp
            (HLO_OrOp
              (HLO_CompareOp (HLO_SubOp:$frac
                               $input,
                               (HLO_FloorOp:$floor $input)),
                             (HLO_ConstOp $half),
                             HLO_COMPARISON_DIRECTION_GT,
                             $compare_type0),
              (HLO_AndOp
                (HLO_CompareOp
                  $frac1,
                  (HLO_ConstOp $half1),
                  HLO_COMPARISON_DIRECTION_EQ,
                  $compare_type1),
                (HLO_CompareOp
                  (HLO_SubOp
                    $floor1,
                    (HLO_MulOp
                      (HLO_FloorOp (HLO_MulOp $input, (HLO_ConstOp $half2))),
                      (HLO_ConstOp $two))),
                  (HLO_ConstOp $one1),
                  HLO_COMPARISON_DIRECTION_EQ,
                  $compare_type2))),
            (HLO_AddOp $floor2, (HLO_ConstOp $one)),
            $floor3),
          (TF_RoundOp $input),
          [(FloatValueEquals<"1.0"> $one),
           (FloatValueEquals<"1.0"> $one1),
           (FloatValueEquals<"2.0"> $two),
           (FloatValueEquals<"0.5"> $half),
           (FloatValueEquals<"0.5"> $half1),
           (FloatValueEquals<"0.5"> $half2),
           (SameValue $floor, $floor1),
           (SameValue $floor, $floor2),
           (SameValue $floor, $floor3),
           (SameValue $frac, $frac1),
           (FloatOrDefaultCompare $compare_type0),
           (FloatOrDefaultCompare $compare_type1),
           (FloatOrDefaultCompare $compare_type2)]>;
