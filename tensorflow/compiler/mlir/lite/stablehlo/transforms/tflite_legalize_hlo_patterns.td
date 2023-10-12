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

// This is the legalization pattern definition file for HLO to TFL.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mhlo/IR/hlo_ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

def CreateTFLCastToInt32Op : NativeCodeCall<
  "CreateCastToInt32($0, $_loc, $_builder)">;
// TODO: b/304003568 - Add TF_TransposeOp folding logic to tflite.
def LegalizeTranspose : Pat<(MHLO_TransposeOp $arg, $perm),
                            (TFL_TransposeOp $arg,
                            (CreateTFLCastToInt32Op (TFL_ConstOp $perm)))>;


def ConvertDotGeneralOp : NativeCodeCall<"ConvertDotGeneralOp($_builder, "
                                               "$0.getDefiningOp())">;
def LegalizeDotGeneral: Pat<(MHLO_DotGeneralOp:$old_value
               $lhs,
               $rhs,
               $dot_dimension_numbers, $precision_config),
          (ConvertDotGeneralOp $old_value)>;