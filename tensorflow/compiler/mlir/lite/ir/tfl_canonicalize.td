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

// This is the optimization pattern definition file for TensorFlow Lite.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/utils/utils.td"

// Returns Squeezed shape of a ranked-tensor.
// Squeezed, here, means eliminating any 1s' in the
// dimensions of the tensor
def GetSqueezedShape: NativeCodeCall<"GetSqueezedShape($0)">;

// This is a utility function to deduct the effective permutation to apply on
// TFL_TransposeOp when the tensor has some dimensions with value==1
def GetSqueezedPermutation: NativeCodeCall<"GetSqueezedPermutation($0, $1)">;

// Check to see if the tensor dimensions can be Squeezed by eliminating 1s'
def CanSqueezeTensor : Constraint<CPred<
  "GetShapeAttr($0).getNumElements() > GetSqueezedShape($0).getNumElements()">>;


// Pattern to convert TFL_TransposeOp with rank>6 to rank<=6 if there are
// redundant dimensions in the tensor. For example- [2x1x3] == [2x3] and 1 is
// not contributing to the dimentionality. This will run if the rank>6
// Pattern will convert-
// %0 = "tfl.transpose"(%arg0, %cst) : (tensor<56x8x56x1x1x1x7xf32>, tensor<7xi32>) -> tensor<1x1x8x56x56x7x1xf32>
// to-
// %0 = "tfl.reshape"(%arg0, %cst) : (tensor<56x8x56x1x1x1x7xf32>, tensor<4xi32>) -> tensor<56x8x56x7xf32>
// %1 = "tfl.transpose"(%0, %cst_0) : (tensor<56x8x56x7xf32>, tensor<4xi32>) -> tensor<8x56x56x7xf32>
// %2 = "tfl.reshape"(%1, %cst_1) : (tensor<8x56x56x7xf32>, tensor<7xi32>) -> tensor<1x1x8x56x56x7x1xf32>
def ConvertTransposeToDecreaseRank : Pat<
  (TFL_TransposeOp:$output_transpose $input, (Arith_ConstantOp:$permutation $_)),
  (TFL_ReshapeOp
    (TFL_TransposeOp
      (TFL_ReshapeOp $input, (Arith_ConstantOp (GetSqueezedShape $input))),
      (Arith_ConstantOp (GetSqueezedPermutation $input, $permutation))),
    (Arith_ConstantOp (GetShapeAttr $output_transpose))),
    [(AnyStaticShapeTensor $input),
     (HasRankAtLeast<7> $input),
     (CanSqueezeTensor $input)]>;

def RemoveNoopTranspose : Pat<
  (TFL_TransposeOp $input, $perm),
  (replaceWithValue $input),
  [(IsTransposeNoop $perm)]>;