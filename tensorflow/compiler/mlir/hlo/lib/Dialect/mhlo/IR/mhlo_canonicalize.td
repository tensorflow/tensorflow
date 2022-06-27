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
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_utils.td"

def UnaryToBinaryEinsumEq : NativeCodeCall<
  "$_builder.getStringAttr(\",\" + $0.getValue().str())">;

// Convert UnaryEinsumOp to EinsumOp with two operands with redundant first
// operand.
def UnaryEinsumToEinsum : Pat<
  (HLO_UnaryEinsumOp $operand, $equation),
  (HLO_EinsumOp (HLO_ConstantOp (GetScalarOfType<1> $operand)),
                $operand, (UnaryToBinaryEinsumEq $equation))>;

// A dynamic reshape of a dynamic reshape is a dynamic reshape.
def RemoveRedundantDynamicReshape : Pat<
  (HLO_DynamicReshapeOp (HLO_DynamicReshapeOp $operand, $shape1), $shape2),
  (HLO_DynamicReshapeOp $operand, $shape2)>;

// A dynamic broadcast of a dynamic reshape with the same shape operand
// is a dynamic reshape.
def RemoveRedundantDynamicBroadcast : Pat<
  (HLO_DynamicBroadcastInDimOp
    (HLO_DynamicReshapeOp $operand, $shape),
    $shape,
    IdentityBroadcastDims,
    $known_expanding_dimensions,
    $known_nonexpanding_dimensions),
  (HLO_DynamicReshapeOp $operand, $shape)>;


// Convert DPad to Pad if edge_padding_low, edge_padding_high and
// interior_paddin are HLO_ConstantOp
def DPadToPad: Pat<
          (HLO_DynamicPadOp HLO_Tensor:$input,
            HLO_Tensor:$padding_value,
            (HLO_ConstantOp I64ElementsAttr:$edge_padding_low),
            (HLO_ConstantOp I64ElementsAttr:$edge_padding_high),
            (HLO_ConstantOp I64ElementsAttr:$interior_paddin)),
          (HLO_PadOp $input, $padding_value,
            (CastIntElementsAttr $edge_padding_low),
            (CastIntElementsAttr $edge_padding_high),
            (CastIntElementsAttr $interior_paddin))>;

// Convert RealDynamicSliceOp to SliceOp if start_indices, limit_indices and
// strides are HLO_ConstantOp
def RealDSliceToSlice: Pat<
          (HLO_RealDynamicSliceOp HLO_Tensor:$operand,
            (HLO_ConstantOp I64ElementsAttr:$start_indices),
            (HLO_ConstantOp I64ElementsAttr:$limit_indices),
            (HLO_ConstantOp I64ElementsAttr:$strides)),
          (HLO_SliceOp $operand,
            (CastIntElementsAttr $start_indices),
            (CastIntElementsAttr $limit_indices),
            (CastIntElementsAttr $strides))>;
