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

// This is the legalization pattern definition file for MHLO to StandardOps.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/Dialect/Math/IR/MathOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"

//===----------------------------------------------------------------------===//
// Nullary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(HLO_ConstantOp ElementsAttr:$value),
          (Arith_ConstantOp $value)>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

def IsSameSizePred : CPred<
    "$0.getType().cast<ShapedType>().getShape() "
    "== $1.getType().cast<ShapedType>().getShape()">;
def IsSameSizeConstraint : Constraint<IsSameSizePred, "inputs are same size">;


// Unary Lowering Patterns.
def : Pat<(HLO_CeilOp HLO_FpTensor:$i), (Math_CeilOp $i)>;

// Binary Lowering Patterns.
def : Pat<(HLO_AndOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_AndIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_OrOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_OrIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_AddOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_AddFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_SubtractOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_SubFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_MulOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_MulFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_DivOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_DivFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_RemOp HLO_FpTensor:$l, HLO_FpTensor:$r),
          (Arith_RemFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_AddOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_AddIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_SubtractOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_SubIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_MulOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_MulIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_DivOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_DivSIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_RemOp HLO_IntTensor:$l, HLO_IntTensor:$r),
          (Arith_RemSIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;

def : Pat<(HLO_SelectOp HLO_PredTensor:$pred, HLO_Tensor:$tv, HLO_Tensor:$fv),
          (SelectOp $pred, $tv, $fv),
          [(IsSameSizeConstraint $pred, $tv), (IsSameSizeConstraint $tv, $fv)]>;
