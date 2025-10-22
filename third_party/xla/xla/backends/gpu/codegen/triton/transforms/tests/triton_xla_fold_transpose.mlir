// RUN: xla-opt %s --triton-xla-fold-transpose | FileCheck %s

// CHECK-LABEL: func @fold_transpose_of_extract
func.func @fold_transpose_of_extract(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<4x8x2xf32> {
  // CHECK: %[[EXTRACT:.*]] = triton_xla.extract from %arg0
  // CHECK-SAME: as memref<4x1x8x2xf32, #triton_xla.layout<[2, 0, 1, 3]>>
  // CHECK-SAME: [0, 0, 0, 0] [4, 1, 8, 2] [1, 1, 1, 1] : tensor<4x8x2xf32>
  %0 = triton_xla.extract from %arg0
    as memref<2x1x4x8xf32, #triton_xla.layout<[3, 2, 1, 0]>>
    [0, 0, 0, 0] [2, 1, 4, 8] [1, 1, 1, 1] : tensor<2x4x8xf32>
  %1 = tt.trans %0 {order = array<i32: 1, 2, 0>} : tensor<2x4x8xf32> -> tensor<4x8x2xf32>
  // CHECK: return %[[EXTRACT]] : tensor<4x8x2xf32>
  return %1 : tensor<4x8x2xf32>
}

// CHECK-LABEL: func @fold_insert_of_transpose
func.func @fold_insert_of_transpose(%arg0: tensor<4x8x2xf32>, %arg1: !tt.ptr<f32>) {
  // CHECK: triton_xla.insert %arg0 into %arg1
  // CHECK-SAME: as memref<4x1x8x2xf32, #triton_xla.layout<[2, 0, 1, 3]>>
  // CHECK-SAME: [0, 0, 0, 0] [4, 1, 8, 2] [1, 1, 1, 1]
  %0 = tt.trans %arg0 {order = array<i32: 2, 0, 1>} : tensor<4x8x2xf32> -> tensor<2x4x8xf32>
  triton_xla.insert %0 into %arg1
    as memref<2x1x4x8xf32, #triton_xla.layout<[3, 2, 1, 0]>>
    [0, 0, 0, 0] [2, 1, 4, 8] [1, 1, 1, 1] : tensor<2x4x8xf32>
  return
}

// CHECK-LABEL: func @push_transpose_up_through_broadcast
func.func @push_transpose_up_through_broadcast(%arg0: tensor<4x1xi1>) -> tensor<8x4xi1> {
  // CHECK: %[[TRANS:.*]] = tt.trans %arg0 {order = array<i32: 1, 0>} : tensor<4x1xi1> -> tensor<1x4xi1>
  // CHECK: tt.broadcast %[[TRANS]] : tensor<1x4xi1> -> tensor<8x4xi1>
  %0 = tt.broadcast %arg0 : tensor<4x1xi1> -> tensor<4x8xi1>
  %1 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<4x8xi1> -> tensor<8x4xi1>
  return %1 : tensor<8x4xi1>
}

// CHECK-LABEL: func @push_transpose_up_through_expand_dims
func.func @push_transpose_up_through_expand_dims(%arg0: tensor<4x8xi1>) -> tensor<1x8x4xi1> {
  // CHECK: %[[TRANS:.*]] = tt.trans %arg0 {order = array<i32: 1, 0>} : tensor<4x8xi1> -> tensor<8x4xi1>
  // CHECK: tt.expand_dims %[[TRANS]] {axis = 0 : i32} : tensor<8x4xi1> -> tensor<1x8x4xi1>
  %0 = tt.expand_dims %arg0 {axis = 1 : i32} : tensor<4x8xi1> -> tensor<4x1x8xi1>
  %1 = tt.trans %0 {order = array<i32: 1, 2, 0>} : tensor<4x1x8xi1> -> tensor<1x8x4xi1>
  return %1 : tensor<1x8x4xi1>
}

// CHECK-LABEL: func @push_transpose_up_through_elementwise
func.func @push_transpose_up_through_elementwise(%arg0: tensor<4x8xf32>) -> tensor<8x4xf32> {
  // CHECK: arith.negf {{.*}} : tensor<8x4xf32>
  %0 = arith.negf %arg0 : tensor<4x8xf32>
  %1 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// CHECK-LABEL: func @push_transpose_up_through_if
func.func @push_transpose_up_through_if(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %cond: i1) -> tensor<8x4xf32> {
  // CHECK-DAG: %[[TRANS0:.*]] = tt.trans %arg0 {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  // CHECK-DAG: %[[TRANS1:.*]] = tt.trans %arg1 {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  %0 = scf.if %cond -> tensor<4x8xf32> {
    // CHECK: scf.yield %[[TRANS0]] : tensor<8x4xf32>
    scf.yield %arg0 : tensor<4x8xf32>
  } else {
    // CHECK: scf.yield %[[TRANS1]] : tensor<8x4xf32>
    scf.yield %arg1 : tensor<4x8xf32>
  }
  // CHECK-NOT: tt.trans
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
