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
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Math/IR/MathOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mhlo/IR/hlo_ops.td"

//===----------------------------------------------------------------------===//
// Nullary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(MHLO_ConstantOp ElementsAttr:$value),
          (Arith_ConstantOp $value)>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

def IsSameSizePred : CPred<
    "$0.getType().cast<ShapedType>().getShape() "
    "== $1.getType().cast<ShapedType>().getShape()">;
def IsSameSizeConstraint : Constraint<IsSameSizePred, "inputs are same size">;
def createFastMathNone : NativeCodeCall<
    "::mlir::arith::FastMathFlagsAttr::get("
      "$_builder.getContext(), ::mlir::arith::FastMathFlags::none"
    ")">;


// Unary Lowering Patterns.
def : Pat<(MHLO_CeilOp MHLO_FpTensor:$i), (Math_CeilOp $i, (createFastMathNone ))>;

// Binary Lowering Patterns.
def : Pat<(MHLO_AndOp MHLO_IntTensor:$l, MHLO_IntTensor:$r),
          (Arith_AndIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_OrOp MHLO_IntTensor:$l, MHLO_IntTensor:$r),
          (Arith_OrIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_AddOp MHLO_FpTensor:$l, MHLO_FpTensor:$r),
          (Arith_AddFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_SubtractOp MHLO_FpTensor:$l, MHLO_FpTensor:$r),
          (Arith_SubFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_MulOp MHLO_FpTensor:$l, MHLO_FpTensor:$r),
          (Arith_MulFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_DivOp MHLO_FpTensor:$l, MHLO_FpTensor:$r),
          (Arith_DivFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_RemOp MHLO_FpTensor:$l, MHLO_FpTensor:$r),
          (Arith_RemFOp $l, $r, (createFastMathNone )),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_AddOp MHLO_IntTensor:$l, MHLO_IntTensor:$r),
          (Arith_AddIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_SubtractOp MHLO_IntTensor:$l, MHLO_IntTensor:$r),
          (Arith_SubIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_MulOp MHLO_IntTensor:$l, MHLO_IntTensor:$r),
          (Arith_MulIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_DivOp MHLO_IntTensor:$l, MHLO_IntTensor:$r),
          (Arith_DivSIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(MHLO_RemOp MHLO_IntTensor:$l, MHLO_IntTensor:$r),
          (Arith_RemSIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;

def : Pat<(MHLO_SelectOp $pred, $tv, $fv),
          (SelectOp $pred, $tv, $fv),
          [(IsSameSizeConstraint $pred, $tv), (IsSameSizeConstraint $tv, $fv)]>;
