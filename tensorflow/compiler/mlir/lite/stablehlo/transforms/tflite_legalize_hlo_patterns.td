/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "mhlo/IR/hlo_ops.td"


def ShapeToConst : NativeCodeCall<"ShapeToConst($_builder, $0)">;

def CreateTFLCastToInt32Op : NativeCodeCall<
  "CreateCastToInt32($0, $_loc, $_builder)">;

def : Pat<
  (MHLO_ConstantOp:$output $value),
  (Arith_ConstantOp $value),
  [(TFL_TensorOf<[AnyType]> $output)]>;


//===----------------------------------------------------------------------===//
// data movement
//===----------------------------------------------------------------------===//

def : Pat<
  (MHLO_TransposeOp $arg, $perm),
  (TFL_TransposeOp $arg,
    (CreateTFLCastToInt32Op
      (TFL_ConstOp $perm)))>;

def : Pat<
  (MHLO_ReverseOp $input, $dims),
  (TFL_ReverseV2Op $input,
    (CreateTFLCastToInt32Op
      (Arith_ConstantOp $dims)))>;

def I64AttrToI32Attr:
  NativeCodeCall<
    "$_builder.getI32IntegerAttr("
      "static_cast<int32_t>($0.cast<IntegerAttr>().getInt()))">;

def : Pat<
  (MHLO_ConcatenateOp $inputs, $dim),
  (TFL_ConcatenationOp
    $inputs,
    (I64AttrToI32Attr $dim),
    TFL_AF_None)>;

//
// Reshapes
//

def : Pat<
  (MHLO_ReshapeOp:$output $input),
  (TFL_ReshapeOp $input,
    (CreateTFLCastToInt32Op
      (ShapeToConst $output)))>;

def : Pat<
  (MHLO_DynamicReshapeOp $input, $shape),
  (TFL_ReshapeOp $input,
    (CreateTFLCastToInt32Op $shape))>;

//
// Broadcasts
//

def IsTFLStyleBroadcast :
  Constraint<CPred<"IsTFLStyleBroadcast($0, $1)">>;

def IsNotTFLStyleBroadcast :
  Constraint<Neg<IsTFLStyleBroadcast.predicate>>;

def ExpandedShape :
  NativeCodeCall<"ExpandedShape($_builder, $0, $1, $2)">;

def ExpandedDynamicShape :
  NativeCodeCall<"ExpandedDynamicShape($_builder, $0, $1, $2)">;

def : Pat<
  (MHLO_BroadcastInDimOp:$output
    $input,
    $broadcast_dimensions),
  (TFL_BroadcastToOp
    $input,
    (ShapeToConst $output)),
  [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>;

def : Pat<
  (MHLO_BroadcastInDimOp:$output
    $input,
    $broadcast_dimensions),
  (TFL_BroadcastToOp
    (TFL_ReshapeOp
        $input,
        (ExpandedShape
          $input,
          $broadcast_dimensions,
          $output)),
    (ShapeToConst $output)),
  [(IsNotTFLStyleBroadcast $broadcast_dimensions, $output)]>;

def : Pat<
  (MHLO_DynamicBroadcastInDimOp:$output
    $input,
    $output_dimensions,
    $broadcast_dimensions,
    $expanding_dimensions_unused,
    $nonexpanding_dimensions_unused),
  (TFL_BroadcastToOp
    $input,
    $output_dimensions),
  [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>;

def : Pat<
  (MHLO_DynamicBroadcastInDimOp:$output
    $input,
    $output_dimensions,
    $broadcast_dimensions,
    $expanding_dimensions_unused,
    $nonexpanding_dimensions_unused),
  (TFL_BroadcastToOp
    (ExpandedDynamicShape
      $input,
      $broadcast_dimensions,
      $output),
    $output_dimensions),
  [(IsNotTFLStyleBroadcast $broadcast_dimensions, $output)]>;



//===----------------------------------------------------------------------===//
// logical and bitwise ops
//===----------------------------------------------------------------------===//

class GetRankedScalarAttr<string prefix, int width, string signed, string value> :
  NativeCodeCall<"DenseElementsAttr::get<" # prefix # "int" # width # "_t>("
    "RankedTensorType::get({}, $_builder.getIntegerType("
      # width # signed # "))," # value # ")">;

def : Pat<(MHLO_NotOp I1Tensor:$input), (TFL_LogicalNotOp $input)>;

// TFL does not support bitwise negation. not(x) is equivalant to xor(x, y) if
// y has a 1 in every bit position (xor(1, 1) = 0 and xor(0, 1) = 1).

// Signed: The 2s complement of -1 has a 1 in every bit position.
def : Pat<(MHLO_NotOp I8Tensor:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"", 8, "", "-1">)))>;

def : Pat<(MHLO_NotOp I16Tensor:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"", 16, "", "-1">)))>;

def : Pat<(MHLO_NotOp I32Tensor:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"", 32, "", "-1">)))>;


// Unsigned: 0xFFF... has a 1 in every bit position.
def : Pat<(MHLO_NotOp TensorOf<[UI8]>:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"u", 8, ", false", "0xFFU">)))>;

def : Pat<(MHLO_NotOp TensorOf<[UI16]>:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"u", 16, ", false", "0xFFFFU">)))>;

def : Pat<(MHLO_NotOp TensorOf<[UI32]>:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"u", 32, ", false", "0xFFFFFFFFUL">)))>;

foreach pair = [
  [MHLO_AndOp, TFL_LogicalAndOp],
  [MHLO_OrOp, TFL_LogicalOrOp],
] in {
  def : Pat<
    (pair[0] TFL_BoolTensor:$l, TFL_BoolTensor:$r),
    (pair[1] $l, $r)>;
}

def LegalizeXor : Pat<
  (MHLO_XorOp
    TFL_IntTensor:$l,
    TFL_IntTensor:$r),
  (TFL_BitwiseXorOp $l, $r)>;

//===----------------------------------------------------------------------===//
// binary element-wise ops
//===----------------------------------------------------------------------===//

def : Pat<
  (MHLO_ShiftRightArithmeticOp $l, $r),
  (TFL_RightShiftOp $l, $r)>;

def : Pat<
  (MHLO_ShiftRightLogicalOp $l, $r),
  (TFL_RightShiftOp $l, $r)>;

def : Pat<
  (MHLO_Atan2Op $l, $r),
  (TFL_Atan2Op $l, $r)>;

def : Pat<
  (MHLO_RemOp $l, $r),
  (TFL_FloorModOp $l, $r)>;

// Binary ops with no attrs.
foreach pair = [
  [MHLO_MaxOp, TFL_MaximumOp],
  [MHLO_MinOp, TFL_MinimumOp],
  [MHLO_PowOp, TFL_PowOp],
] in {
  def : Pat<
    (pair[0] $l, $r),
    (pair[1] $l, $r)>; 

  def : Pat<
    (pair[0] 
      (MHLO_BroadcastInDimOp:$output
        $bcast_operand,
        $broadcast_dimensions),
      $r),
    (pair[1] $bcast_operand, $r),
    [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>; 

  def : Pat<
    (pair[0] 
      $l,
      (MHLO_BroadcastInDimOp:$output
        $bcast_operand,
        $broadcast_dimensions)),
    (pair[1] $l, $bcast_operand),
    [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>; 
}

// Binary ops with fused activiation attr.
foreach pair = [
  [MHLO_DivOp, TFL_DivOp],
  [MHLO_AddOp, TFL_AddOp],
  [MHLO_MulOp, TFL_MulOp],
  [MHLO_SubtractOp, TFL_SubOp],
] in {
  def : Pat<
    (pair[0] $l, $r),
    (pair[1] $l, $r, TFL_AF_None)>;

  def : Pat<
    (pair[0] 
      (MHLO_BroadcastInDimOp:$output
        $bcast_operand,
        $broadcast_dimensions),
      $r),
    (pair[1] $bcast_operand, $r, TFL_AF_None),
    [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>; 

  def : Pat<
    (pair[0] 
      $l,
      (MHLO_BroadcastInDimOp:$output
        $bcast_operand,
        $broadcast_dimensions)),
    (pair[1] $l, $bcast_operand, TFL_AF_None),
    [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>; 
}



//===----------------------------------------------------------------------===//
// comparison ops
//===----------------------------------------------------------------------===//

// Check implicit bool cast of `$_self` to ensure Attribute is non-null before
// casting.
def HasSupportedComparisonType : AttrConstraint<
  CPred<"!$_self || SupportedComparisonType($_self.cast<mhlo::ComparisonTypeAttr>())">>;

class MHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<MHLO_ComparisonDirectionAttr,
    "::mlir::mhlo::ComparisonDirection::" # enumStr>;

foreach p = [
  [TFL_EqualOp,        MHLO_ComparisonDirectionValue<"EQ">],
  [TFL_NotEqualOp,     MHLO_ComparisonDirectionValue<"NE">],
  [TFL_GreaterEqualOp, MHLO_ComparisonDirectionValue<"GE">],
  [TFL_LessEqualOp,    MHLO_ComparisonDirectionValue<"LE">],
  [TFL_GreaterOp,      MHLO_ComparisonDirectionValue<"GT">],
  [TFL_LessOp,         MHLO_ComparisonDirectionValue<"LT">]]
in {
  def : Pat<
    (MHLO_CompareOp $l, $r, p[1], HasSupportedComparisonType),
    (p[0] $l, $r)>;
}

//===----------------------------------------------------------------------===//
// unary element-wise op
//===----------------------------------------------------------------------===//

def LowerCbrt : Pat<(MHLO_CbrtOp $opr),
  (TFL_PowOp $opr,
    (TFL_DivOp
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_None)),
      [(F32Tensor $opr)]>;


foreach pair = [
  [MHLO_AbsOp, TFL_AbsOp],
  [MHLO_BitcastConvertOp, TFL_BitcastOp],
  [MHLO_CeilOp, TFL_CeilOp],
  [MHLO_CosineOp, TFL_CosOp],
  [MHLO_ExpOp, TFL_ExpOp],
  [MHLO_FloorOp, TFL_FloorOp],
  [MHLO_ImagOp, TFL_ImagOp],
  [MHLO_LogOp, TFL_LogOp],
  [MHLO_LogisticOp, TFL_LogisticOp],
  [MHLO_NegOp, TFL_NegOp],
  [MHLO_RealOp, TFL_RealOp],
  [MHLO_RsqrtOp, TFL_RsqrtOp],
  [MHLO_SineOp, TFL_SinOp],
  [MHLO_SignOp, TFL_SignOp],
  [MHLO_SqrtOp, TFL_SqrtOp],
  [MHLO_TanhOp, TFL_TanhOp]
] in {
  def : Pat<
    (pair[0] $input),
    (pair[1] $input)>;
}

def : Pat<
  (MHLO_ConvertOp $input),
  (TFL_CastOp $input)>;

def : Pat<
  (MHLO_Expm1Op F32Tensor:$x),
  (TFL_SubOp
    (TFL_ExpOp $x),
    (Arith_ConstantOp
      ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
    TFL_AF_None)>;

def : Pat<
  (MHLO_IsFiniteOp F32Tensor:$x),
  (TFL_EqualOp
    (TFL_SubOp $x, $x, TFL_AF_None),
    (Arith_ConstantOp
      ConstantAttr<RankedF32ElementsAttr<[]>, "0.0f">))>;

def : Pat<
  (MHLO_Log1pOp F32Tensor:$x),
  (TFL_LogOp
    (TFL_AddOp
      $x,
      (Arith_ConstantOp
        ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
      TFL_AF_None))>;

//===----------------------------------------------------------------------===//
// rounding
//===----------------------------------------------------------------------===//

class ValueEquals<string val> :
  Constraint<CPred<"ValueEquals($0, " # val # ")">>;

def SameValue :
  Constraint<CPred<"$0 == $1">>;

def FloatOrDefaultCompare :
  Constraint<CPred<"!$0 || $0.getValue() == ::mlir::mhlo::ComparisonType::FLOAT">>;

def SameTypeOrDefaultCompare :
  Constraint<CPred<"SameTypeOrDefaultCompare($0, $1)">>;

def ValueIsReciprocal :
  Constraint<CPred<"ValueIsReciprocal($0, $1)">>;

def TensorIsSign :
  Constraint<CPred<"TensorIsSign($_builder, $0, $1)">>;

def ValueGreaterThanZero :
  Constraint<CPred<"ValueGreaterThanZero($0)">>;

def : Pat<
  (MHLO_RoundNearestEvenOp $input),
  (TFL_RoundOp $input)>;

// Converts a dag of HLOs representing banker rounding (round x.5 to nearest
// even) to tfl.round. This only supports float types because mhlo.floor only
// supports float types. tf.round with integer input type will become an
// identity op, so we will never face an mhlo.floor with an integer input type.
// The pattern matched executes the following computation:
// frac = x - floor(x)
// to_even = (floor(x) - 2 * floor(0.5 * x)) == 1
// if frac > 0.5 || (frac == 0.5 && to_even)
//   return floor(x) + 1
// else
//   return floor(x)
def Phase1_Round : Pat<(MHLO_SelectOp
            (MHLO_OrOp
              (MHLO_CompareOp (MHLO_SubtractOp:$frac
                               $input,
                               (MHLO_FloorOp:$floor $input)),
                             (MHLO_ConstantOp $half),
                             MHLO_ComparisonDirectionValue<"GT">,
                             $compare_type0),
              (MHLO_AndOp
                (MHLO_CompareOp
                  $frac1,
                  (MHLO_ConstantOp $half1),
                  MHLO_ComparisonDirectionValue<"EQ">,
                  $compare_type1),
                (MHLO_CompareOp
                  (MHLO_SubtractOp
                    $floor1,
                    (MHLO_MulOp
                      (MHLO_FloorOp (MHLO_MulOp $input, (MHLO_ConstantOp $half2))),
                      (MHLO_ConstantOp $two))),
                  (MHLO_ConstantOp $one1),
                  MHLO_ComparisonDirectionValue<"EQ">,
                  $compare_type2))),
            (MHLO_AddOp $floor2, (MHLO_ConstantOp $one)),
            $floor3),
          (TFL_RoundOp $input),
          [(ValueEquals<"1.0"> $one),
           (ValueEquals<"1.0"> $one1),
           (ValueEquals<"2.0"> $two),
           (ValueEquals<"0.5"> $half),
           (ValueEquals<"0.5"> $half1),
           (ValueEquals<"0.5"> $half2),
           (SameValue $floor, $floor1),
           (SameValue $floor, $floor2),
           (SameValue $floor, $floor3),
           (SameValue $frac, $frac1),
           (FloatOrDefaultCompare $compare_type0),
           (FloatOrDefaultCompare $compare_type1),
           (FloatOrDefaultCompare $compare_type2)]>;

// Converts a dag of HLOs representing floor_mod to tfl.floor_mod.
// The pattern matched executes the following computation:
//
// rem = remainder(arg0, arg1)
// for i in 0 to len(arg1):
//    if ((rem[i] < 0) != (arg0[i] < 0) && arg0[i] != 0)
//       rem[i] += arg1[i]
// return rem
def Phase1_FloorMod : Pat<(MHLO_SelectOp
            (MHLO_AndOp
              (MHLO_CompareOp
                (MHLO_CompareOp:$rltz
                  (MHLO_RemOp:$rem $arg, $arg1),
                  (MHLO_ConstantOp $cst),
                  MHLO_ComparisonDirectionValue<"LT">,
                  $compare_type),
                (MHLO_CompareOp:$arg1ltz $arg1, (MHLO_ConstantOp $cst1), MHLO_ComparisonDirectionValue<"LT">, $compare_type1),
                MHLO_ComparisonDirectionValue<"NE">,
                $compare_type2),
              (MHLO_CompareOp:$rnz $rem1, (MHLO_ConstantOp $cst2), MHLO_ComparisonDirectionValue<"NE">, $compare_type3)),
            (MHLO_AddOp $rem2, $arg1),
            $rem3),
          (TFL_FloorModOp $arg, $arg1),
          [(ValueEquals<"0.0"> $cst),
           (ValueEquals<"0.0"> $cst1),
           (ValueEquals<"0.0"> $cst2),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (SameValue $rem, $rem3),
           (SameTypeOrDefaultCompare $compare_type, $cst),
           (SameTypeOrDefaultCompare $compare_type1, $cst1)]>;

// Converts a dag of HLOs representing floor_mod with a constant to
// tfl.floor_mod. The pattern matched executes the following computation:
//
// cst = value that is > 0
// rem = remainder(arg0, cst)
// for i in 0 to len(arg1):
//    if (rem[i] < 0 && rem[i] != 0)
//       rem[i] += cst
// return rem
def Phase1_FloorMod2 : Pat<(MHLO_SelectOp
            (MHLO_AndOp
              (MHLO_CompareOp:$rltz
                (MHLO_RemOp:$rem $arg, (MHLO_ConstantOp $cst)),
                (MHLO_ConstantOp $cst1),
                MHLO_ComparisonDirectionValue<"LT">,
                $compare_type),
              (MHLO_CompareOp:$rnz $rem1, (MHLO_ConstantOp $cst2), MHLO_ComparisonDirectionValue<"NE">, $compare_type3)),
            (MHLO_AddOp $rem2, (MHLO_ConstantOp $cst3)),
            $rem3),
          (TFL_FloorModOp $arg, (Arith_ConstantOp $cst3)),
          [(ValueGreaterThanZero $cst),
           (ValueEquals<"0.0"> $cst1),
           (ValueEquals<"0.0"> $cst2),
           (SameValue $cst, $cst3),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (SameValue $rem, $rem3),
           (SameTypeOrDefaultCompare $compare_type, $cst1),
           (SameTypeOrDefaultCompare $compare_type3, $cst2)]>;

// Converts a dag of HLOs representing floor_div to tfl.floor_div.
// The pattern matched executes the following computation:
//
// rem = remainder(arg0, arg1)
// for i in 0 to len(arg1):
//    rem[i] = arg0[i] - rem[i] / arg1[i]
//    if (rem[i] != 0 && sign(arg1[i]) != sign(rem[i]))
//       rem[i] -= 1.0
// return round_nearest_afz(rem)
// As a dag this looks like the following:
//                       round
//                         |
//              -------- select
//              |          |    \
//             &&          +    div
//           /   |        / \
//        !=    !=       div -1
//       / |   / |      / |
//    rem 0.0 sn sn1    -  $1
//   / |      |  |    / |
// $0 $1     $1 rem  $0 rem
// Note that named operators like 'sn' and 'sn1' are different values produced by
// the same function in this case the sign function. Named values like 'div'
// refer to the same value produced by the same function, in this case division.
// Mathematical symbols do not indicate a re-use of the value.
def Phase1_FloorDiv : Pat<(MHLO_RoundOp
            (MHLO_SelectOp
              (MHLO_AndOp
                (MHLO_CompareOp
                  (MHLO_RemOp:$rem $arg0, $arg1),
                  (MHLO_ConstantOp $cst),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (MHLO_CompareOp
                  (MHLO_SignOp $arg1),
                  (MHLO_SignOp $rem1),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (MHLO_AddOp
                (MHLO_DivOp:$div
                  (MHLO_SubtractOp $arg0, $rem2),
                  $arg1b),
                (MHLO_ConstantOp $cst_neg1)),
              $div1)),
          (TFL_FloorDivOp $arg0, $arg1),
          [(ValueEquals<"0.0"> $cst),
           (ValueEquals<"-1.0"> $cst_neg1),
           (SameValue $div, $div1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (FloatOrDefaultCompare $compare_type, $cst),
           (FloatOrDefaultCompare $compare_type1, $cst)]>;

// Converts a dag of HLOs representing floor_div with a splat constant to
// tfl.floor_div. The pattern matched executes the following computation:
// This particular pattern matches multiplication with the reciprocal of the
// constant instead of dividing by the constant.
// rem = remainder(arg0, cst)
// for i in 0 to len(arg0):
//    rem[i] = (arg0[i] - rem[i]) * 1 / cst
//    if (rem[i] != 0 && sign(cst) != sign(rem[i]))
//       rem[i] += -1.0
// return round_nearest_afz(rem)
// As a dag this looks like the following:
//                       round
//                         |
//              -------- select
//              |           |     \
//             &&           +     mul
//           /   |         / \
//        !=    !=        mul -1
//       / |   /  |       / |
//    rem 0.0 cs1 sn1   -  cs2
//   / |           |   / |
// $0 cst         rem $0 rem
// cs1 == sign(cst)
// cs2 = 1 / cst i.e. the reciprocal
// Note that named operators like 'sn' and 'sn1' are different values produced by
// the same function in this case the sign function. Named values like 'div'
// refer to the same value produced by the same function, in this case division.
// Mathematical symbols do not indicate a re-use of the value.
def Phase1_FloorDiv2 : Pat<(MHLO_RoundOp
            (MHLO_SelectOp
              (MHLO_AndOp
                (MHLO_CompareOp
                  (MHLO_RemOp:$rem $arg0, (MHLO_ConstantOp:$cst $cstv)),
                  (MHLO_ConstantOp $cst_zero),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (MHLO_CompareOp
                  (MHLO_ConstantOp $cst_sgn),
                  (MHLO_SignOp $rem1),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (MHLO_AddOp
                (MHLO_MulOp:$mul
                  (MHLO_SubtractOp $arg0, $rem2),
                  (MHLO_ConstantOp $cst_recip)),
                (MHLO_ConstantOp $cst_neg1)),
              $mul1)),
          (TFL_FloorDivOp $arg0, $cst),
          [(ValueEquals<"0.0"> $cst_zero),
           (ValueEquals<"-1.0"> $cst_neg1),
           (TensorIsSign $cstv, $cst_sgn),
           (ValueIsReciprocal $cstv, $cst_recip),
           (SameValue $mul, $mul1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (FloatOrDefaultCompare $compare_type, $cst_zero),
           (FloatOrDefaultCompare $compare_type1, $cst_sgn)]>;

// Converts a dag of HLOs representing floor_div with a splat constant to
// tfl.floor_div. The pattern matched executes the following computation:
// This particular pattern matches division with the constant.
// .
// rem = remainder(arg0, cst)
// for i in 0 to len(arg0):
//    rem[i] = (arg0[i] - rem[i]) / cst
//    if (rem[i] != 0 && sign(cst) != sign(rem[i]))
//       rem[i] -= 1.0
// return round_nearest_afz(rem)
// As a dag this looks like the following:
//                       round
//                         |
//              -------- select
//              |           |     \
//             &&           +     div
//           /   |         / \
//        !=    !=        div -1
//       / |   /  |       / |
//    rem 0.0 cs1 sn1   -  cs2
//   / |           |   / |
// $0 cst         rem $0 rem
// cs1 == sign(cst)
// cs2 = 1 / cst i.e. the reciprocal
// Note that named operators like 'sn' and 'sn1' are different values produced by
// the same function in this case the sign function. Named values like 'div'
// refer to the same value produced by the same function, in this case division.
// Mathematical symbols do not indicate a re-use of the value.
def Phase1_FloorDiv3 : Pat<(MHLO_RoundOp
            (MHLO_SelectOp
              (MHLO_AndOp
                (MHLO_CompareOp
                  (MHLO_RemOp:$rem $arg0, (MHLO_ConstantOp:$cst $cstv)),
                  (MHLO_ConstantOp $cst_zero),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (MHLO_CompareOp
                  (MHLO_ConstantOp $cst_sgn),
                  (MHLO_SignOp $rem1),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (MHLO_AddOp
                (MHLO_DivOp:$div
                  (MHLO_SubtractOp $arg0, $rem2),
                  (MHLO_ConstantOp $cstv1)),
                (MHLO_ConstantOp $cst_neg1)),
              $div1)),
          (TFL_FloorDivOp $arg0, $cst),
          [(ValueEquals<"0.0"> $cst_zero),
           (ValueEquals<"-1.0"> $cst_neg1),
           (TensorIsSign $cstv, $cst_sgn),
           (SameValue $div, $div1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (SameValue $cstv1, $cstv),
           (FloatOrDefaultCompare $compare_type, $cst_zero),
           (FloatOrDefaultCompare $compare_type1, $cst_sgn)]>;

// Converts a dag of HLOs representing floor_div with a broadcasted vector
// constant to tfl.floor_div. The pattern matched executes the following
// computation:
// scs = sign(cst)
// bcst = broadcast(cst)
// rem = remainder(arg0, bcst)
// for i in 0 to len(arg0):
//    rem[i] = arg0[i] - rem[i] * / bcst
//    if (rem[i] != 0 && scs != sign(rem[i]))
//       rem[i] -= 1.0
// return round_nearest_afz(rem)
// Where scs is a splat constant folded sign on the unbroadcasted tensor.
//
// As a dag this looks like the following:
//                       round
//                         |
//              -------- select
//              |           |     \
//             &&           +     div
//           /   |         / \
//        !=    !=        div -1
//       / |   /  |       / |
//    rem 0.0 scs sn1   -  bcst
//   / |           |   / |
// $0 bcst        rem $0 rem
//     |
//    cst
// scs == sign(cst) == sign(bcst)
// Note that named operators like 'sn' and 'sn1' are different values produced by
// the same function in this case the sign function. Named values like 'div'
// refer to the same value produced by the same function, in this case division.
// Mathematical symbols do not indicate a re-use of the value.
def Phase1_FloorDiv4 : Pat<(MHLO_RoundOp
            (MHLO_SelectOp
              (MHLO_AndOp
                (MHLO_CompareOp
                  (MHLO_RemOp:$rem $arg0,
                    (MHLO_BroadcastInDimOp:$bcst
                      (MHLO_ConstantOp $cstv),
                      $broadcast_dimension)),
                  (MHLO_ConstantOp $cst_zero),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (MHLO_CompareOp
                  (MHLO_ConstantOp $cst_sgn),
                  (MHLO_SignOp $rem1),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (MHLO_AddOp
                (MHLO_DivOp:$div
                  (MHLO_SubtractOp $arg0, $rem2),
                  $bcst1),
                (MHLO_ConstantOp $cst_neg1)),
              $div1)),
          (TFL_FloorDivOp $arg0, $bcst),
          [(ValueEquals<"0.0"> $cst_zero),
           (ValueEquals<"-1.0"> $cst_neg1),
           (TensorIsSign $cstv, $cst_sgn),
           (SameValue $bcst, $bcst1),
           (SameValue $div, $div1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (FloatOrDefaultCompare $compare_type, $cst_zero),
           (FloatOrDefaultCompare $compare_type1, $cst_sgn)]>;

// Semantically equivalant to above, just uses sub(x, 1) instead of
// add(x, -1).
def Phase1_FloorDiv5 : Pat<
          (MHLO_RoundOp
            (MHLO_SelectOp
              (MHLO_AndOp
                (MHLO_CompareOp
                  (MHLO_RemOp:$rem
                    $arg0,
                    (MHLO_ConstantOp:$cstv_op $cstv)),
                  (MHLO_ConstantOp $cst_zero),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (MHLO_CompareOp
                  (MHLO_ConstantOp $cst_sgn),
                  (MHLO_SignOp $rem1),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (MHLO_SubtractOp
                (MHLO_DivOp:$div
                  (MHLO_SubtractOp $arg0, $rem2),
                  $cstv_op_2),
                (MHLO_ConstantOp $cst_1)),
              $div1)),
          (TFL_FloorDivOp $arg0, $cstv_op),
          [(ValueEquals<"0.0"> $cst_zero),
           (ValueEquals<"1.0"> $cst_1),
           (TensorIsSign $cstv, $cst_sgn),
           (SameValue $cstv_op, $cstv_op_2),
           (SameValue $div, $div1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (FloatOrDefaultCompare $compare_type, $cst_zero),
           (FloatOrDefaultCompare $compare_type1, $cst_sgn)]>;



//===----------------------------------------------------------------------===//
// ternary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(MHLO_ClampOp $min, $arg, $max),
          (TFL_MaximumOp (TFL_MinimumOp $arg, $max), $min)>;
def : Pat<(MHLO_SelectOp $cond, $t, $e), (TFL_SelectOp $cond, $t, $e)>;
