// RUN: xla-opt %s -split-input-file \
// RUN: -stablehlo-lower-transpose-to-triton \
// RUN: | FileCheck %s

// CHECK-LABEL: func @lower_transpose(%arg0: tensor<2x4x8xf32>) -> tensor<8x2x4xf32>
func.func @lower_transpose(%arg0: tensor<2x4x8xf32>) -> tensor<8x2x4xf32> {
  // CHECK: %0 = tt.trans %arg0 {order = array<i32: 2, 0, 1>} : tensor<2x4x8xf32> -> tensor<8x2x4xf32>
  %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<2x4x8xf32>) -> tensor<8x2x4xf32>
  // CHECK: return %0 : tensor<8x2x4xf32>
  return %0 : tensor<8x2x4xf32>
}
