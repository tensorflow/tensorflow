// RUN: xla-opt %s --triton-xla-fold-transpose | FileCheck %s

// CHECK-LABEL: func @fold_transpose_of_load_ptr
tt.func @fold_transpose_of_load_ptr(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<8x4xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  // CHECK:      %[[PTR:.*]] = tt.make_tensor_ptr %arg0,
  // CHECK-SAME:     [%c8_i64, %c4_i64], [%c1_i64, %c8_i64], [%c0_i32, %c0_i32]
  // CHECK-SAME:     {order = array<i32: 1, 0>} : <tensor<8x4xf32>>
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

// CHECK-LABEL: func @push_transpose_up_through_elementwise
tt.func @push_transpose_up_through_elementwise(%arg0: tensor<4x8xf32>) -> tensor<8x4xf32> {
  // CHECK: arith.negf {{.*}} : tensor<8x4xf32>
  %0 = arith.negf %arg0 : tensor<4x8xf32>
  %1 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  tt.return %1 : tensor<8x4xf32>
}

// CHECK-LABEL: func @push_transpose_up_through_reshape
tt.func @push_transpose_up_through_reshape(%arg0: tensor<4x8x2xf32>) -> tensor<16x4xf32> {
  // CHECK: tt.reshape {{.*}} : tensor<8x2x4xf32> -> tensor<16x4xf32>
  %0 = tt.reshape %arg0 : tensor<4x8x2xf32> -> tensor<4x16xf32>
  %1 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<4x16xf32> -> tensor<16x4xf32>
  tt.return %1 : tensor<16x4xf32>
}

// CHECK-LABEL: func @push_transpose_up_through_join_of_inline_asm
tt.func @push_transpose_up_through_join_of_inline_asm(%arg0: tensor<4x8xf32>) -> tensor<8x4x2xf32> {
  // CHECK: tt.elementwise_inline_asm {{.*}} : tensor<8x4xf32> -> tensor<8x4xf32>, tensor<8x4xf32>
  %0:2 = tt.elementwise_inline_asm "" {constraints = "", packed_element = 1 : i32, pure = true} %arg0 : tensor<4x8xf32> -> tensor<4x8xf32>, tensor<4x8xf32>
  // CHECK: tt.join {{.*}} : tensor<8x4xf32> -> tensor<8x4x2xf32>
  %1 = tt.join %0#0, %0#1 : tensor<4x8xf32> -> tensor<4x8x2xf32>
  %2 = tt.trans %1 {order = array<i32: 1, 0, 2>} : tensor<4x8x2xf32> -> tensor<8x4x2xf32>
  tt.return %2 : tensor<8x4x2xf32>
}

// CHECK-LABEL: func @push_transpose_up_through_int4_unpack
tt.func @push_transpose_up_through_int4_unpack(%arg0: tensor<8x4xi8>) -> tensor<4x16xbf16> {
  // CHECK: tt.trans {{.*}} : tensor<8x4xi8> -> tensor<4x8xi8>
  // CHECK-NOT: tt.trans
  %0:2 = tt.elementwise_inline_asm "" {constraints = "", packed_element = 1 : i32, pure = true} %arg0 : tensor<8x4xi8> -> tensor<8x4xbf16>, tensor<8x4xbf16>
  %1 = tt.join %0#0, %0#1 : tensor<8x4xbf16> -> tensor<8x4x2xbf16>
  %2 = tt.trans %1 {order = array<i32: 0, 2, 1>} : tensor<8x4x2xbf16> -> tensor<8x2x4xbf16>
  %3 = tt.reshape %2 : tensor<8x2x4xbf16> -> tensor<16x4xbf16>
  %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<16x4xbf16> -> tensor<4x16xbf16>
  tt.return %4 : tensor<4x16xbf16>
}
