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
include "mlir/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/xla/ir/xla_ops.td"

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

def IsSameSizePred : CPred<
    "$0->getType().cast<ShapedType>().getShape() "
    "== $1->getType().cast<ShapedType>().getShape()">;
def IsSameSizeConstraint : Constraint<IsSameSizePred, "inputs are same size">;

def : Pat<(XLA_AddOp XLA_FpTensor:$l, XLA_FpTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (AddFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(XLA_SubOp XLA_FpTensor:$l, XLA_FpTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (SubFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(XLA_MulOp XLA_FpTensor:$l, XLA_FpTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (MulFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(XLA_DivOp XLA_FpTensor:$l, XLA_FpTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (DivFOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;

def : Pat<(XLA_AddOp XLA_IntTensor:$l, XLA_IntTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (AddIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(XLA_SubOp XLA_IntTensor:$l, XLA_IntTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (SubIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(XLA_MulOp XLA_IntTensor:$l, XLA_IntTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (MulIOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;
def : Pat<(XLA_DivOp XLA_IntTensor:$l, XLA_IntTensor:$r,
                     IsNullAttr:$broadcast_dimensions),
          (DivISOp $l, $r),
          [(IsSameSizeConstraint $l, $r)]>;

