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

// This is the legalization pattern definition file for XLA to StandardOps.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops.td"

//===----------------------------------------------------------------------===//
// Nullary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(HLO_ConstOp ElementsAttr:$value),
          (ConstantOp $value)>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

def IsSameSizePred : CPred<
    "$0->getType().cast<ShapedType>().getShape() "
    "== $1->getType().cast<ShapedType>().getShape()">;
def IsSameSizeConstraint : Constraint<IsSameSizePred, "inputs are same size">;


def : Pat<(HLO_AndOp HLO_PredTensor:$l, HLO_PredTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (AndOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_AddOp HLO_FpTensor:$l, HLO_FpTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (AddFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_SubOp HLO_FpTensor:$l, HLO_FpTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (SubFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_MulOp HLO_FpTensor:$l, HLO_FpTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (MulFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_DivOp HLO_FpTensor:$l, HLO_FpTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (DivFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_RemOp HLO_FpTensor:$l, HLO_FpTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (RemFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_AddOp HLO_IntTensor:$l, HLO_IntTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (AddIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_SubOp HLO_IntTensor:$l, HLO_IntTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (SubIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_MulOp HLO_IntTensor:$l, HLO_IntTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (MulIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_DivOp HLO_IntTensor:$l, HLO_IntTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (DivISOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(HLO_RemOp HLO_IntTensor:$l, HLO_IntTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (RemISOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
