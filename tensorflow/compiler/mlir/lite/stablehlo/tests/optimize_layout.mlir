// RUN: odml-to-stablehlo-opt %s --transpose-commute-ops | FileCheck %s
// CHECK-LABEL:   func.func @commute_transpose_pad(
// CHECK-SAME:          %[[INPUT:.*]]: tensor<1x112x112x64xf32>,
// CHECK-SAME:          %[[PAD_VAL:.*]]: tensor<f32>) -> tensor<1x64x114x114xf32> {
// CHECK:           %[[PAD:.*]] = stablehlo.pad %[[INPUT]], %[[PAD_VAL]],
// CHECK:               low = [0, 1, 1, 0], high = [0, 1, 1, 0], interior = [0, 0, 0, 0]
// CHECK:               : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<1x114x114x64xf32>
// CHECK:           %[[TPOS:.*]] = stablehlo.transpose %[[PAD]], dims = [0, 3, 1, 2]
// CHECK:               : (tensor<1x114x114x64xf32>) -> tensor<1x64x114x114xf32>
// CHECK:           return %[[TPOS]] : tensor<1x64x114x114xf32>

func.func @commute_transpose_pad(
      %arg0: tensor<1x112x112x64xf32>, %padding_val: tensor<f32>)
      -> tensor<1x64x114x114xf32> {
  %tspos = stablehlo.transpose %arg0, dims = [0, 3, 1, 2]
      : (tensor<1x112x112x64xf32>) -> tensor<1x64x112x112xf32>
  %ret = stablehlo.pad %tspos, %padding_val,
      low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 0, 0]
      : (tensor<1x64x112x112xf32>, tensor<f32>) -> tensor<1x64x114x114xf32>
  return %ret :tensor<1x64x114x114xf32>
}

// -----
// CHECK-LABEL:   func.func @commute_transpose_reduce_window(
// CHECK-SAME:          %[[INPUT:.*]]: tensor<1x114x114x64xf32>,
// CHECK-SAME:          %[[PAD_VAL:.*]]: tensor<f32>) -> tensor<1x64x56x56xf32> {
// CHECK:           %[[REDUCE:.*]] = "stablehlo.reduce_window"(%[[INPUT]], %[[PAD_VAL]])
// CHECK:               <{window_dimensions = array<i64: 1, 3, 3, 1>,
// CHECK:                 window_strides = array<i64: 1, 2, 2, 1>}> ({
// CHECK:           ^bb0(%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>):
// CHECK:             %[[MAX:.*]] = stablehlo.maximum %[[ARG0]], %[[ARG1]] : tensor<f32>
// CHECK:             stablehlo.return %[[MAX]] : tensor<f32>
// CHECK:           }) : (tensor<1x114x114x64xf32>, tensor<f32>) -> tensor<1x56x56x64xf32>
// CHECK:           %[[TPOS:.*]] = stablehlo.transpose %[[REDUCE]], dims = [0, 3, 1, 2]
// CHECK:              : (tensor<1x56x56x64xf32>) -> tensor<1x64x56x56xf32>
// CHECK:           return %[[TPOS]] : tensor<1x64x56x56xf32>

func.func @commute_transpose_reduce_window(
      %input: tensor<1x114x114x64xf32>,
      %cst: tensor<f32>) -> tensor<1x64x56x56xf32> {
  %tpos = stablehlo.transpose %input, dims = [0, 3, 1, 2]
      : (tensor<1x114x114x64xf32>) -> tensor<1x64x114x114xf32>
  %ret = "stablehlo.reduce_window"(%tpos, %cst)
      <{window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    stablehlo.return %max: tensor<f32>
  }) : (tensor<1x64x114x114xf32>, tensor<f32>) -> tensor<1x64x56x56xf32>
  return %ret : tensor<1x64x56x56xf32>
}
