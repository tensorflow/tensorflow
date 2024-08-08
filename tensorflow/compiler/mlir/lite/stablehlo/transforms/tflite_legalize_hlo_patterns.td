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
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

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


//===----------------------------------------------------------------------===//
// comparison ops
//===----------------------------------------------------------------------===//

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
  (MHLO_CompareOp $l, $r, p[1], HasSupportedComparisonType:$type),
  (p[0] $l, $r)>;
}
