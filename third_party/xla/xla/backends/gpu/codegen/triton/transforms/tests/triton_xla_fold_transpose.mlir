// RUN: xla-opt %s --triton-xla-fold-transpose | FileCheck %s

// CHECK-LABEL: func @fold_transpose_of_extract
func.func @fold_transpose_of_extract(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<8x4xf32> {
  // CHECK: %[[EXTRACT_0:.*]] = triton_xla.extract from %arg0 as memref<8x4xf32,
  // CHECK-SAME: #triton_xla.layout<[1, 0]>> [0, 0] [8, 4] [1, 1] : tensor<8x4xf32>
  %0 = triton_xla.extract from %arg0 
    as memref<4x8xf32, #triton_xla.layout<[0, 1]>>
    [0, 0] [4, 8] [1, 1] : tensor<4x8xf32>
  %1 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  // CHECK: return %[[EXTRACT_0]] : tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// CHECK-LABEL: func @push_transpose_up_through_elementwise
func.func @push_transpose_up_through_elementwise(%arg0: tensor<4x8xf32>) -> tensor<8x4xf32> {
  // CHECK: arith.negf {{.*}} : tensor<8x4xf32>
  %0 = arith.negf %arg0 : tensor<4x8xf32>
  %1 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// CHECK-LABEL: func @push_transpose_up_through_reshape
func.func @push_transpose_up_through_reshape(%arg0: tensor<4x8x2xf32>) -> tensor<16x4xf32> {
  // CHECK: tt.reshape {{.*}} : tensor<8x2x4xf32> -> tensor<16x4xf32>
  %0 = tt.reshape %arg0 : tensor<4x8x2xf32> -> tensor<4x16xf32>
  %1 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<4x16xf32> -> tensor<16x4xf32>
  return %1 : tensor<16x4xf32>
}
