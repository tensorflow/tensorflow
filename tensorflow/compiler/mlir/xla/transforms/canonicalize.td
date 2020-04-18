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

def UnaryToBinaryEinsumEq : NativeCodeCall<
  "$_builder.getStringAttr(\",\" + $0.getValue().str())">;

// Convert UnaryEinsumOp to EinsumOp with two operands with redundant first
// operand.
def UnaryEinsumToEinsum : Pat<
  (HLO_UnaryEinsumOp $operand, $equation),
  (HLO_EinsumOp (HLO_ConstOp (GetScalarOfType<1> $operand)),
                $operand, (UnaryToBinaryEinsumEq $equation))>;

//===----------------------------------------------------------------------===//
// Conv op patterns.
//===----------------------------------------------------------------------===//

def IsZero : Attr<CPred<
  "($_self.isa<DenseFPElementsAttr>() &&"
  "$_self.cast<DenseFPElementsAttr>().isSplat() &&"
  "$_self.cast<DenseFPElementsAttr>().getSplatValue<FloatAttr>()"
  ".getValue().isZero()) ||"
  "($_self.isa<DenseIntElementsAttr>() &&"
  "$_self.cast<DenseIntElementsAttr>().isSplat() &&"
  "$_self.cast<DenseIntElementsAttr>().getSplatValue<IntegerAttr>()"
  ".getInt() == 0)">>;

def IsOnlyPaddingSpatialDims
  : Constraint<CPred<"IsOnlyPaddingSpatialDims($0, $1, $2, $3)">>;

def BuildConvPaddingAttrs : NativeCodeCall<
  "BuildConvPaddingAttrs($0, $1, $2, $3, &$_builder)">;

def FoldPadIntoConv : Pat<
  (HLO_ConvOp
    (HLO_PadOp $lhs,
      (HLO_ConstOp IsZero:$padding_value),
      $edge_padding_low,
      $edge_padding_high,
      IsZero:$interior_padding),
    $rhs,
    $window_strides,
    $padding,
    $lhs_dilation,
    $rhs_dilation,
    $dimension_numbers,
    $feature_group_count,
    $batch_group_count,
    $precision_config),
  (HLO_ConvOp
    $lhs,
    $rhs,
    $window_strides,
    (BuildConvPaddingAttrs $edge_padding_low, $edge_padding_high, $padding,
      $dimension_numbers),
    $lhs_dilation,
    $rhs_dilation,
    $dimension_numbers,
    $feature_group_count,
    $batch_group_count,
    $precision_config),
    [(IsOnlyPaddingSpatialDims $lhs, $dimension_numbers, $edge_padding_low,
      $edge_padding_high)]>;
