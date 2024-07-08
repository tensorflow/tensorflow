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
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mhlo/IR/hlo_ops.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"


def IsNotWithinScatterRegion : Constraint<CPred<"!llvm::isa<mhlo::ScatterOp>($0.getDefiningOp()->getParentOp())">,
    "binary ops within scatter regions are not converted">;

def ShapeToConst : NativeCodeCall<"ShapeToConst($_builder, $0)">;

def CreateTFLCastToInt32Op : NativeCodeCall<
  "CreateCastToInt32($0, $_loc, $_builder)">;

def LegalizeTranspose : Pat<(MHLO_TransposeOp $arg, $perm),
                            (TFL_TransposeOp $arg,
                            (CreateTFLCastToInt32Op (TFL_ConstOp $perm)))>;

def LowerCbrt : Pat<(MHLO_CbrtOp $opr),
  (TFL_PowOp $opr,
    (TFL_DivOp
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_None)),
      [(F32Tensor $opr)]>;

def LegalizeReshape : Pat<(MHLO_ReshapeOp:$output $input),
          (TFL_ReshapeOp $input, 
            (CreateTFLCastToInt32Op (ShapeToConst $output)))>;

def LegalizeDynamicReshape : Pat<(MHLO_DynamicReshapeOp $input, $shape),
          (TFL_ReshapeOp $input, (CreateTFLCastToInt32Op $shape))>;


foreach fromToBinPair = [[MHLO_MaxOp, TFL_MaximumOp],
                         [MHLO_MinOp, TFL_MinimumOp],
                         [MHLO_PowOp, TFL_PowOp],
                         [MHLO_Atan2Op, TFL_Atan2Op]] in {
  def : Pat<(fromToBinPair[0]:$result $l, $r),
            (fromToBinPair[1] $l, $r),
            // TODO: b/351437662 - Determine if this is still needed.
            [(IsNotWithinScatterRegion $result)]>;
}

foreach fromToBinPairWithFaf = [[MHLO_DivOp, TFL_DivOp],
                         [MHLO_AddOp, TFL_AddOp],
                         [MHLO_MulOp, TFL_MulOp],
                         [MHLO_SubtractOp, TFL_SubOp]] in {
  def : Pat<(fromToBinPairWithFaf[0]:$result $l, $r),
            (fromToBinPairWithFaf[1] $l, $r, TFL_AF_None),
            // TODO: b/351437662 - Determine if this is still needed.
            [(IsNotWithinScatterRegion $result)]>;
}