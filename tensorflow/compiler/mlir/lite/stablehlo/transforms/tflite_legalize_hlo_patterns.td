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
include "stablehlo/dialect/StablehloOps.td"


def ShapeToConst : NativeCodeCall<"ShapeToConst($_builder, $0)">;

def CreateTFLCastToInt32Op : NativeCodeCall<
  "CreateCastToInt32($0, $_loc, $_builder)">;

def CreateI32ConstFromI64Array : NativeCodeCall<
  "CreateI32ConstFromI64Array($0, $_loc, $_builder)">;

def IsBuildableWithArithConstant : Constraint<
  CPred<"::mlir::arith::ConstantOp::isBuildableWith($0, $1.getType())">
>;

def : Pat<
  (StableHLO_ConstantOp:$output $value),
  (Arith_ConstantOp $value),
  [(IsBuildableWithArithConstant $value, $output)],
  [],
  (addBenefit 10)>;

def : Pat<
  (StableHLO_ConstantOp:$output $value),
  (TFL_ConstOp $value),
  [(TFL_TensorOf<[AnyType]> $output)]>;


//===----------------------------------------------------------------------===//
// data movement
//===----------------------------------------------------------------------===//

def : Pat<
  (StableHLO_TransposeOp $arg, $perm),
  (TFL_TransposeOp $arg,
    (CreateI32ConstFromI64Array $perm))>;

def : Pat<
  (StableHLO_ReverseOp $input, $dims),
  (TFL_ReverseV2Op $input,
    (CreateI32ConstFromI64Array $dims))>;

def I64AttrToI32Attr:
  NativeCodeCall<
    "$_builder.getI32IntegerAttr("
      "static_cast<int32_t>(llvm::cast<IntegerAttr>($0).getInt()))">;

def : Pat<
  (StableHLO_ConcatenateOp $inputs, $dim),
  (TFL_ConcatenationOp
    $inputs,
    (I64AttrToI32Attr $dim),
    TFL_AF_None)>;

//
// Reshapes
//

def : Pat<
  (StableHLO_ReshapeOp:$output $input),
  (TFL_ReshapeOp $input,
    (CreateTFLCastToInt32Op
      (ShapeToConst $output)))>;

def : Pat<
  (StableHLO_DynamicReshapeOp $input, $shape),
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
  (StableHLO_BroadcastInDimOp:$output
    $input,
    $broadcast_dimensions),
  (TFL_BroadcastToOp
    $input,
    (ShapeToConst $output)),
  [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>;

def : Pat<
  (StableHLO_BroadcastInDimOp:$output
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
  (StableHLO_DynamicBroadcastInDimOp:$output
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
  (StableHLO_DynamicBroadcastInDimOp:$output
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

def : Pat<(StableHLO_NotOp I1Tensor:$input), (TFL_LogicalNotOp $input)>;

// TFL does not support bitwise negation. not(x) is equivalant to xor(x, y) if
// y has a 1 in every bit position (xor(1, 1) = 0 and xor(0, 1) = 1).

// Signed: The 2s complement of -1 has a 1 in every bit position.
def : Pat<(StableHLO_NotOp I8Tensor:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"", 8, "", "-1">)))>;

def : Pat<(StableHLO_NotOp I16Tensor:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"", 16, "", "-1">)))>;

def : Pat<(StableHLO_NotOp I32Tensor:$input),
            (TFL_BitwiseXorOp $input,
              (Arith_ConstantOp
                (GetRankedScalarAttr<"", 32, "", "-1">)))>;


// Unsigned: 0xFFF... has a 1 in every bit position.
def : Pat<(StableHLO_NotOp TensorOf<[UI8]>:$input),
            (TFL_BitwiseXorOp $input,
              (TFL_ConstOp
                (GetRankedScalarAttr<"u", 8, ", false", "0xFFU">)))>;

def : Pat<(StableHLO_NotOp TensorOf<[UI16]>:$input),
            (TFL_BitwiseXorOp $input,
              (TFL_ConstOp
                (GetRankedScalarAttr<"u", 16, ", false", "0xFFFFU">)))>;

def : Pat<(StableHLO_NotOp TensorOf<[UI32]>:$input),
            (TFL_BitwiseXorOp $input,
              (TFL_ConstOp
                (GetRankedScalarAttr<"u", 32, ", false", "0xFFFFFFFFUL">)))>;

foreach pair = [
  [StableHLO_AndOp, TFL_LogicalAndOp],
  [StableHLO_OrOp, TFL_LogicalOrOp],
] in {
  def : Pat<
    (pair[0] TFL_BoolTensor:$l, TFL_BoolTensor:$r),
    (pair[1] $l, $r)>;
}

def LegalizeXor : Pat<
  (StableHLO_XorOp
    TFL_IntTensor:$l,
    TFL_IntTensor:$r),
  (TFL_BitwiseXorOp $l, $r)>;

//===----------------------------------------------------------------------===//
// binary element-wise ops
//===----------------------------------------------------------------------===//

def : Pat<
  (StableHLO_ShiftRightArithmeticOp $l, $r),
  (TFL_RightShiftOp $l, $r)>;

def : Pat<
  (StableHLO_ShiftRightLogicalOp $l, $r),
  (TFL_RightShiftOp $l, $r)>;

def : Pat<
  (StableHLO_Atan2Op $l, $r),
  (TFL_Atan2Op $l, $r)>;

def : Pat<
  (StableHLO_RemOp $l, $r),
  (TFL_FloorModOp $l, $r)>;

// Binary ops with no attrs.
foreach pair = [
  [StableHLO_MaxOp, TFL_MaximumOp],
  [StableHLO_MinOp, TFL_MinimumOp],
  [StableHLO_PowOp, TFL_PowOp],
] in {
  def : Pat<
    (pair[0] $l, $r),
    (pair[1] $l, $r)>; 

  def : Pat<
    (pair[0] 
      (StableHLO_BroadcastInDimOp:$output
        $bcast_operand,
        $broadcast_dimensions),
      $r),
    (pair[1] $bcast_operand, $r),
    [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>; 

  def : Pat<
    (pair[0] 
      $l,
      (StableHLO_BroadcastInDimOp:$output
        $bcast_operand,
        $broadcast_dimensions)),
    (pair[1] $l, $bcast_operand),
    [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>; 
}

// Binary ops with fused activiation attr.
foreach pair = [
  [StableHLO_DivOp, TFL_DivOp],
  [StableHLO_AddOp, TFL_AddOp],
  [StableHLO_MulOp, TFL_MulOp],
  [StableHLO_SubtractOp, TFL_SubOp],
] in {
  def : Pat<
    (pair[0] $l, $r),
    (pair[1] $l, $r, TFL_AF_None)>;

  def : Pat<
    (pair[0] 
      (StableHLO_BroadcastInDimOp:$output
        $bcast_operand,
        $broadcast_dimensions),
      $r),
    (pair[1] $bcast_operand, $r, TFL_AF_None),
    [(IsTFLStyleBroadcast $broadcast_dimensions, $output)]>; 

  def : Pat<
    (pair[0] 
      $l,
      (StableHLO_BroadcastInDimOp:$output
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
  CPred<"!$_self || SupportedComparisonType(llvm::cast<stablehlo::ComparisonTypeAttr>($_self))">>;

class StableHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<StableHLO_ComparisonDirectionAttr,
    "::mlir::stablehlo::ComparisonDirection::" # enumStr>;

foreach p = [
  [TFL_EqualOp,        StableHLO_ComparisonDirectionValue<"EQ">],
  [TFL_NotEqualOp,     StableHLO_ComparisonDirectionValue<"NE">],
  [TFL_GreaterEqualOp, StableHLO_ComparisonDirectionValue<"GE">],
  [TFL_LessEqualOp,    StableHLO_ComparisonDirectionValue<"LE">],
  [TFL_GreaterOp,      StableHLO_ComparisonDirectionValue<"GT">],
  [TFL_LessOp,         StableHLO_ComparisonDirectionValue<"LT">]]
in {
  def : Pat<
    (StableHLO_CompareOp $l, $r, p[1], HasSupportedComparisonType),
    (p[0] $l, $r)>;
}

//===----------------------------------------------------------------------===//
// unary element-wise op
//===----------------------------------------------------------------------===//

def LowerCbrt : Pat<(StableHLO_CbrtOp $opr, $accuracy),
  (TFL_PowOp $opr,
    (TFL_DivOp
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_None)),
      [(F32Tensor $opr)]>;

// Pattern to legalize stablehlo.abs(complex) to tfl.complex_abs.
def : Pat<(StableHLO_AbsOp HLO_ComplexTensor:$arg), (TFL_ComplexAbsOp $arg)>;

// Pattern to match non-complex abs.
def : Pat<(StableHLO_AbsOp HLO_PredIntFpOrQuantizedTensor:$arg), (TFL_AbsOp $arg)>;

foreach pair = [
  [StableHLO_BitcastConvertOp, TFL_BitcastOp],
  [StableHLO_CeilOp, TFL_CeilOp],
  [StableHLO_FloorOp, TFL_FloorOp],
  [StableHLO_ImagOp, TFL_ImagOp],
  [StableHLO_NegOp, TFL_NegOp],
  [StableHLO_RealOp, TFL_RealOp],
  [StableHLO_SignOp, TFL_SignOp],
] in {
  def : Pat<
    (pair[0] $input),
    (pair[1] $input)>;
}

foreach pair = [
  [StableHLO_CosineOp, TFL_CosOp],
  [StableHLO_ExpOp, TFL_ExpOp],
  [StableHLO_LogOp, TFL_LogOp],
  [StableHLO_LogisticOp, TFL_LogisticOp],
  [StableHLO_RsqrtOp, TFL_RsqrtOp],
  [StableHLO_SineOp, TFL_SinOp],
  [StableHLO_SqrtOp, TFL_SqrtOp],
  [StableHLO_TanhOp, TFL_TanhOp],
] in {
  def : Pat<
    (pair[0] $input, $accuracy),
    (pair[1] $input)>;
}


def : Pat<
  (StableHLO_ConvertOp $input),
  (TFL_CastOp $input)>;

def : Pat<
  (StableHLO_Expm1Op F32Tensor:$x, $accuracy),
  (TFL_SubOp
    (TFL_ExpOp $x),
    (Arith_ConstantOp
      ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
    TFL_AF_None)>;

def : Pat<
  (StableHLO_IsFiniteOp F32Tensor:$x),
  (TFL_EqualOp
    (TFL_SubOp $x, $x, TFL_AF_None),
    (Arith_ConstantOp
      ConstantAttr<RankedF32ElementsAttr<[]>, "0.0f">))>;

def : Pat<
  (StableHLO_Log1pOp F32Tensor:$x, $accuracy),
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
  Constraint<CPred<"!$0 || $0.getValue() == ::mlir::stablehlo::ComparisonType::FLOAT">>;

def SameTypeOrDefaultCompare :
  Constraint<CPred<"SameTypeOrDefaultCompare($0, $1)">>;

def ValueIsReciprocal :
  Constraint<CPred<"ValueIsReciprocal($0, $1)">>;

def TensorIsSign :
  Constraint<CPred<"TensorIsSign($_builder, $0, $1)">>;

def ValueGreaterThanZero :
  Constraint<CPred<"ValueGreaterThanZero($0)">>;

def : Pat<
  (StableHLO_RoundNearestEvenOp $input),
  (TFL_RoundOp $input)>;

// Converts a dag of HLOs representing banker rounding (round x.5 to nearest
// even) to tfl.round. This only supports float types because stablehlo.floor only
// supports float types. tf.round with integer input type will become an
// identity op, so we will never face an stablehlo.floor with an integer input type.
// The pattern matched executes the following computation:
// frac = x - floor(x)
// to_even = (floor(x) - 2 * floor(0.5 * x)) == 1
// if frac > 0.5 || (frac == 0.5 && to_even)
//   return floor(x) + 1
// else
//   return floor(x)
def Phase1_Round : Pat<(StableHLO_SelectOp
            (StableHLO_OrOp
              (StableHLO_CompareOp (StableHLO_SubtractOp:$frac
                               $input,
                               (StableHLO_FloorOp:$floor $input)),
                             (StableHLO_ConstantOp $half),
                             StableHLO_ComparisonDirectionValue<"GT">,
                             $compare_type0),
              (StableHLO_AndOp
                (StableHLO_CompareOp
                  $frac1,
                  (StableHLO_ConstantOp $half1),
                  StableHLO_ComparisonDirectionValue<"EQ">,
                  $compare_type1),
                (StableHLO_CompareOp
                  (StableHLO_SubtractOp
                    $floor1,
                    (StableHLO_MulOp
                      (StableHLO_FloorOp (StableHLO_MulOp $input, (StableHLO_ConstantOp $half2))),
                      (StableHLO_ConstantOp $two))),
                  (StableHLO_ConstantOp $one1),
                  StableHLO_ComparisonDirectionValue<"EQ">,
                  $compare_type2))),
            (StableHLO_AddOp $floor2, (StableHLO_ConstantOp $one)),
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
def Phase1_FloorMod : Pat<(StableHLO_SelectOp
            (StableHLO_AndOp
              (StableHLO_CompareOp
                (StableHLO_CompareOp:$rltz
                  (StableHLO_RemOp:$rem $arg, $arg1),
                  (StableHLO_ConstantOp $cst),
                  StableHLO_ComparisonDirectionValue<"LT">,
                  $compare_type),
                (StableHLO_CompareOp:$arg1ltz $arg1, (StableHLO_ConstantOp $cst1), StableHLO_ComparisonDirectionValue<"LT">, $compare_type1),
                StableHLO_ComparisonDirectionValue<"NE">,
                $compare_type2),
              (StableHLO_CompareOp:$rnz $rem1, (StableHLO_ConstantOp $cst2), StableHLO_ComparisonDirectionValue<"NE">, $compare_type3)),
            (StableHLO_AddOp $rem2, $arg1),
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
def Phase1_FloorMod2 : Pat<(StableHLO_SelectOp
            (StableHLO_AndOp
              (StableHLO_CompareOp:$rltz
                (StableHLO_RemOp:$rem $arg, (StableHLO_ConstantOp $cst)),
                (StableHLO_ConstantOp $cst1),
                StableHLO_ComparisonDirectionValue<"LT">,
                $compare_type),
              (StableHLO_CompareOp:$rnz $rem1, (StableHLO_ConstantOp $cst2), StableHLO_ComparisonDirectionValue<"NE">, $compare_type3)),
            (StableHLO_AddOp $rem2, (StableHLO_ConstantOp $cst3)),
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
def Phase1_FloorDiv : Pat<(StableHLO_RoundOp
            (StableHLO_SelectOp
              (StableHLO_AndOp
                (StableHLO_CompareOp
                  (StableHLO_RemOp:$rem $arg0, $arg1),
                  (StableHLO_ConstantOp $cst),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (StableHLO_CompareOp
                  (StableHLO_SignOp $arg1),
                  (StableHLO_SignOp $rem1),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (StableHLO_AddOp
                (StableHLO_DivOp:$div
                  (StableHLO_SubtractOp $arg0, $rem2),
                  $arg1b),
                (StableHLO_ConstantOp $cst_neg1)),
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
// tfl.floor_div.
def Phase1_FloorDiv2 : Pat<(StableHLO_RoundOp
            (StableHLO_SelectOp
              (StableHLO_AndOp
                (StableHLO_CompareOp
                  (StableHLO_RemOp:$rem $arg0, (StableHLO_ConstantOp:$cst $cstv)),
                  (StableHLO_ConstantOp $cst_zero),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (StableHLO_CompareOp
                  (StableHLO_ConstantOp $cst_sgn),
                  (StableHLO_SignOp $rem1),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (StableHLO_AddOp
                (StableHLO_MulOp:$mul
                  (StableHLO_SubtractOp $arg0, $rem2),
                  (StableHLO_ConstantOp $cst_recip)),
                (StableHLO_ConstantOp $cst_neg1)),
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
// tfl.floor_div.
def Phase1_FloorDiv3 : Pat<(StableHLO_RoundOp
            (StableHLO_SelectOp
              (StableHLO_AndOp
                (StableHLO_CompareOp
                  (StableHLO_RemOp:$rem $arg0, (StableHLO_ConstantOp:$cst $cstv)),
                  (StableHLO_ConstantOp $cst_zero),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (StableHLO_CompareOp
                  (StableHLO_ConstantOp $cst_sgn),
                  (StableHLO_SignOp $rem1),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (StableHLO_AddOp
                (StableHLO_DivOp:$div
                  (StableHLO_SubtractOp $arg0, $rem2),
                  (StableHLO_ConstantOp $cstv1)),
                (StableHLO_ConstantOp $cst_neg1)),
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
// constant to tfl.floor_div.
def Phase1_FloorDiv4 : Pat<(StableHLO_RoundOp
            (StableHLO_SelectOp
              (StableHLO_AndOp
                (StableHLO_CompareOp
                  (StableHLO_RemOp:$rem $arg0,
                    (StableHLO_BroadcastInDimOp:$bcst
                      (StableHLO_ConstantOp $cstv),
                      $broadcast_dimension)),
                  (StableHLO_ConstantOp $cst_zero),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (StableHLO_CompareOp
                  (StableHLO_ConstantOp $cst_sgn),
                  (StableHLO_SignOp $rem1),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (StableHLO_AddOp
                (StableHLO_DivOp:$div
                  (StableHLO_SubtractOp $arg0, $rem2),
                  $bcst1),
                (StableHLO_ConstantOp $cst_neg1)),
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
          (StableHLO_RoundOp
            (StableHLO_SelectOp
              (StableHLO_AndOp
                (StableHLO_CompareOp
                  (StableHLO_RemOp:$rem
                    $arg0,
                    (StableHLO_ConstantOp:$cstv_op $cstv)),
                  (StableHLO_ConstantOp $cst_zero),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (StableHLO_CompareOp
                  (StableHLO_ConstantOp $cst_sgn),
                  (StableHLO_SignOp $rem1),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (StableHLO_SubtractOp
                (StableHLO_DivOp:$div
                  (StableHLO_SubtractOp $arg0, $rem2),
                  $cstv_op_2),
                (StableHLO_ConstantOp $cst_1)),
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

def Phase1_FloorDiv6 : Pat<
          (StableHLO_RoundOp
            (StableHLO_SelectOp
              (StableHLO_AndOp
                (StableHLO_CompareOp
                  (StableHLO_RemOp:$rem
                    $arg0,
                    (StableHLO_BroadcastInDimOp:$bcst
                      (StableHLO_ConstantOp $cstv),
                      $broadcast_dimension)),
                  (StableHLO_BroadcastInDimOp
                    (StableHLO_ConstantOp $cst_zero),
                    $broadcast_dimension_zero),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (StableHLO_CompareOp
                  (StableHLO_BroadcastInDimOp
                    (StableHLO_ConstantOp $cst_sgn),
                    $broadcast_dimension_sgn),
                  (StableHLO_SignOp $rem1),
                  StableHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (StableHLO_SubtractOp
                (StableHLO_DivOp:$div
                  (StableHLO_SubtractOp $arg0, $rem2),
                  $bcst1),
                (StableHLO_BroadcastInDimOp
                  (StableHLO_ConstantOp $cst_1),
                  $broadcast_dimension_one)),
              $div1)),
          (TFL_FloorDivOp $arg0, $bcst),
          [(ValueEquals<"0.0"> $cst_zero),
           (ValueEquals<"1.0"> $cst_1),
           (TensorIsSign $cstv, $cst_sgn),
           (SameValue $bcst, $bcst1),
           (SameValue $div, $div1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (FloatOrDefaultCompare $compare_type, $cst_zero),
           (FloatOrDefaultCompare $compare_type1, $cst_sgn)]>;



//===----------------------------------------------------------------------===//
// ternary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(StableHLO_ClampOp $min, $arg, $max),
          (TFL_MaximumOp (TFL_MinimumOp $arg, $max), $min)>;
def : Pat<(StableHLO_SelectOp $cond, $t, $e), (TFL_SelectOp $cond, $t, $e)>;

