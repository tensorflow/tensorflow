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
include "tensorflow/compiler/mlir/xla/ir/hlo_ops.td"

def SignedIntTensor : TensorOf<[I1, I8, I16, I32, I64]>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

class DirectBinaryPat<Op FromOp, Op ToOp>
  : Pat<(FromOp $l, $r, $_), (ToOp $l, $r)>;

foreach fromToBinPair = [[HLO_AddOp, TF_AddV2Op],
                         [HLO_DivOp, TF_DivOp],
                         [HLO_ShiftLeftOp, TF_LeftShiftOp],
                         [HLO_MaxOp, TF_MaximumOp],
                         [HLO_MinOp, TF_MinimumOp],
                         [HLO_MulOp, TF_MulOp],
                         [HLO_PowOp, TF_PowOp],
                         [HLO_DivOp, TF_RealDivOp],
                         [HLO_SubOp, TF_SubOp]] in
  def : DirectBinaryPat<fromToBinPair[0], fromToBinPair[1]>;

def LowerRightShiftSigned :
  Pat<(HLO_ShiftRightArithmeticOp $l, $r, $_), (TF_RightShiftOp $l, $r),
      [(SignedIntTensor $r)]>;

def : Pat<(HLO_FloorOp (HLO_DivOp $l, $r, $_)), (TF_FloorDivOp $l, $r)>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

foreach Mapping = [
                   [HLO_AbsOp, TF_AbsOp],
                   [HLO_CeilOp, TF_CeilOp],
                   [HLO_CosOp, TF_CosOp],
                   [HLO_ExpOp, TF_ExpOp],
                   [HLO_FloorOp, TF_FloorOp],
                   [HLO_ImagOp, TF_ImagOp],
                   [HLO_IsFiniteOp, TF_IsFiniteOp],
                   [HLO_LogOp, TF_LogOp],
                   [HLO_Log1pOp, TF_Log1pOp],
                   [HLO_NotOp, TF_LogicalNotOp],
                   [HLO_NegOp, TF_NegOp],
                   [HLO_RealOp, TF_RealOp],
                   [HLO_RsqrtOp, TF_RsqrtOp],
                   [HLO_SinOp, TF_SinOp],
                   [HLO_SqrtOp, TF_SqrtOp],
                   [HLO_TanhOp, TF_TanhOp],
                  ] in {
 def : Pat<(Mapping[0] $input), (Mapping[1] $input)>;
}
