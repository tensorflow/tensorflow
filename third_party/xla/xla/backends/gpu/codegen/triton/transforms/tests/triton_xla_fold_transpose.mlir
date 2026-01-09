// RUN: xla-opt %s --triton-xla-fold-transpose | FileCheck %s

// CHECK-LABEL: func @push_transpose_of_extract_tile_to_memref
// CHECK-SAME: (%[[INPUT:.*]]: memref
// CHECK-SAME: , %[[OFFSET0:.*]]: index, %[[OFFSET1:.*]]: index, %[[OFFSET2:.*]]: index)
func.func @push_transpose_of_extract_tile_to_memref(
  %input: memref<4x8x16xf32, #xtile.layout<[2, 0, 1]>>,
  %offset0: index, %offset1: index, %offset2: index)  ->  tensor<8x4xf32>
{
  // CHECK: %[[TRANSPOSE:.*]] = memref.transpose %[[INPUT]]

  // CHECK-SAME: (d0, d1, d2) -> (d2, d1, d0)
  // CHECK-SAME: : memref<4x8x16xf32, #xtile.layout<[2, 0, 1]>>
  // CHECK-SAME: to memref<16x8x4xf32, strided<[1, 64, 16]>>
  // CHECK: %[[EXTRACT:.*]] = xtile.extract %[[TRANSPOSE]]
  // CHECK-SAME: [%[[OFFSET2]], %[[OFFSET1]], %[[OFFSET0]]] [8, 1, 4] [1, 1, 1]
  // CHECK-SAME: : memref<16x8x4xf32, strided<[1, 64, 16]>> -> tensor<8x4xf32>
  %tile = xtile.extract %input[%offset0, %offset1, %offset2][4, 1, 8][1, 1, 1]
    : memref<4x8x16xf32, #xtile.layout<[2, 0, 1]>> -> tensor<4x8xf32>
  %transposed = tt.trans %tile {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  // CHECK: return %[[EXTRACT]] : tensor<8x4xf32>
  return %transposed : tensor<8x4xf32>
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

// CHECK-LABEL: func @push_transpose_up_through_mask
func.func @push_transpose_up_through_mask(%arg0: tensor<4x8xf32>, %arg1: f32) -> tensor<8x4xf32> {
  // CHECK: xtile.mask %{{.*}} bounds [7, 3], %arg1 : tensor<8x4xf32>
  %0 = xtile.mask %arg0 bounds [3, 7], %arg1 : tensor<4x8xf32>
  %1 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}
