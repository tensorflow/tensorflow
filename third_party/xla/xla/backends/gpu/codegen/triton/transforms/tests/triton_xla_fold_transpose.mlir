// RUN: xla-opt %s --triton-xla-fold-transpose | FileCheck %s

// CHECK-LABEL: func @fold_transpose_of_load_ptr
tt.func @fold_transpose_of_load_ptr(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<8x4xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  // CHECK:      %[[PTR:.*]] = tt.make_tensor_ptr %arg0,
  // CHECK-SAME:     [%c8_i64, %c4_i64], [%c1_i64, %c8_i64], [%c0_i32, %c0_i32]
  // CHECK-SAME:     {order = array<i32: 0, 1>} : <tensor<8x4xf32>>
  %0 = tt.make_tensor_ptr %arg0, [%c4_i64, %c8_i64], [%c8_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
  // CHECK:      %[[LOAD:.*]] = tt.load %[[PTR]]
  // CHECK-SAME:     {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<8x4xf32>>
  %1 = tt.load %0 {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<4x8xf32>>
  // CHECK-NOT:  tt.trans
  %2 = tt.trans %1 {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  tt.return %2 : tensor<8x4xf32>
}

// CHECK-LABEL: func @fold_transpose_of_load_ptr_with_mask
tt.func @fold_transpose_of_load_ptr_with_mask(%arg0: !tt.ptr<f32>, %arg1: tensor<4x8xi1>) -> tensor<8x4xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %0 = tt.make_tensor_ptr %arg0, [%c4_i64, %c8_i64], [%c8_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<4x8xf32>>
  // CHECK: tt.load {{.*}}, %arg1 : !tt.ptr<tensor<4x8xf32>>
  %1 = tt.load %0, %arg1 : !tt.ptr<tensor<4x8xf32>>
  // CHECK: tt.trans
  %2 = tt.trans %1 {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  tt.return %2 : tensor<8x4xf32>
}
