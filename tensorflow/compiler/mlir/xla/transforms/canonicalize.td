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

// This is the canonicalize pattern definition file.

include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_utils.td"

//===----------------------------------------------------------------------===//
// DynamicSlice op patterns.
//===----------------------------------------------------------------------===//

def BuildSliceLimits : NativeCodeCall<
    "BuildSliceLimits($0.cast<DenseIntElementsAttr>(),"
    "$1.cast<DenseIntElementsAttr>(), &$_builder)">;

def BuildSliceStrides : NativeCodeCall<
  "GetI64ElementsAttr(SmallVector<int64_t, 4>("
  "$0->getType().cast<RankedTensorType>().getRank(), 1), &$_builder)">;

def DynamicSliceToSlice: Pat<(HLO_DynamicSliceOp HLO_Tensor:$input,
           (HLO_ConstOp I64ElementsAttr:$starting_indices),
           I64ElementsAttr:$slice_sizes),
          (HLO_SliceOp $input, (CastIntElementsAttr $starting_indices),
           (BuildSliceLimits $starting_indices, $slice_sizes),
            (BuildSliceStrides $input))>;

def UnaryToBinaryEinsumEq : NativeCodeCall<
  "$_builder.getStringAttr(\",\" + $0.getValue().str())">;

// Convert UnaryEinsumOp to EinsumOp with two operands with redundant first
// operand.
def UnaryEinsumToEinsum : Pat<
  (HLO_UnaryEinsumOp $operand, $equation),
  (HLO_EinsumOp (HLO_ConstOp (GetScalarOfType<1> $operand)),
                $operand, (UnaryToBinaryEinsumEq $equation))>;
