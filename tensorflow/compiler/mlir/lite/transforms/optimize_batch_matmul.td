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

// Optimization patterns which replaces patterns with BatchMatMul.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/utils/utils.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def NotFromFoldableChain : Constraint<CPred<"NotFromFoldableChain($0)">>;

def IsResultRankEqualTo : Constraint<CPred<
  "llvm::cast<ShapedType>($0.getType().front()).getRank() == "
  "llvm::cast<ShapedType>($1.getType()).getRank()">>;

// Fuses TFL_FullyConnectedOp and TFL_TransposeOp Rhs to TFL_BatchMatMulOp when
// it's used by TFL_BatchMatMulOp and "transpose_lhs" is true.
def FuseTransposeFCLhsToBatchMatmul : Pat<
  (TFL_BatchMatMulOp
    (TFL_FullyConnectedOp:$fc_output
      (TFL_TransposeOp TensorOf<[F32]>:$fc_lhs,
                       (Arith_ConstantOp:$perm_value $p0)),
      TensorOf<[F32]>:$fc_rhs,
      $bias, $TFL_AF_None, $TFL_FCWO_Default,
      $keep_num_dims, $asymmetric_quantize_inputs_fc
    ),
    $bmm_rhs, ConstBoolAttrTrue, $transpose_rhs,
    $asymmetric_quantize_inputs_bmm
  ),
  (TFL_BatchMatMulOp
    (TFL_BatchMatMulOp $fc_rhs, $fc_lhs, ConstBoolAttrFalse, ConstBoolAttrFalse,
                       $asymmetric_quantize_inputs_fc,
                       (returnType (TransposeFCLastTwoDims $fc_output))
    ),
    $bmm_rhs, ConstBoolAttrFalse, $transpose_rhs,
    $asymmetric_quantize_inputs_bmm
  ),
  [(AreLastTwoDimsTransposed $perm_value), (IsNoneType $bias)]>;

// Fuses TFL_FullyConnectedOp and TFL_TransposeOp Rhs to TFL_BatchMatMulOp
def FuseTransposeFCRhsToBatchMatmul : Pat<
  (TFL_FullyConnectedOp:$output
    TensorOf<[F32]>:$lhs,
    (TFL_TransposeOp TensorOf<[F32]>:$rhs, (Arith_ConstantOp:$perm_value $p0)),
    $bias, $TFL_AF_None, $TFL_FCWO_Default,
    $keep_num_dims, $asymmetric_quantize_inputs
    ),
  (TFL_BatchMatMulOp $lhs, $rhs, ConstBoolAttrFalse, ConstBoolAttrFalse,
                     $asymmetric_quantize_inputs),
  [(NotConstantLike $rhs),
   (AreLastTwoDimsTransposed $perm_value),
   (IsNoneType $bias),
   //if the output rank is less-than the input rank, it means keep_num_dims has
   //reduced the output. The right thing to do here will be to apply the pattern
   //only if keep_num_dims==True. But checking for rank will improve the
   //coverage. This pattern will now work for all the transpose->fc patterns
   //where the fc has not done imlicit reshape of the output
   (IsResultRankEqualTo $output, $lhs),
   (NotFromFoldableChain $rhs)]>;
