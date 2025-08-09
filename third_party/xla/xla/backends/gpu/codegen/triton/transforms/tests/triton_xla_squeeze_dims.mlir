// RUN: xla-opt %s --triton-xla-squeeze-dims="finalize=false" \
// RUN: | FileCheck %s

// RUN: xla-opt %s --triton-xla-squeeze-dims \
// RUN: | FileCheck %s --check-prefix=FINALIZE

// CHECK-LABEL: func @push_squeeze_dims_up_through_elementwise
tt.func @push_squeeze_dims_up_through_elementwise(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // CHECK: arith.negf {{.*}} : tensor<4x8xf32>
  %0 = arith.negf %arg0 : tensor<4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @push_squeeze_dims_up_through_multiple_results
tt.func @push_squeeze_dims_up_through_multiple_results(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // CHECK: tt.elementwise_inline_asm {{.*}} : tensor<4x8xf32> -> tensor<4x8xf32>, tensor<4x8xf32>
  %0:2 = tt.elementwise_inline_asm "" {constraints = "", packed_element = 1 : i32, pure = true} %arg0 : tensor<4x1x8xf32> -> tensor<4x1x8xf32>, tensor<4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0#0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @push_squeeze_dims_up_through_broadcast
tt.func @push_squeeze_dims_up_through_broadcast(%arg0: tensor<1x4x1x8xf32>) -> tensor<4x16x8xf32> {
  // CHECK: tt.broadcast {{.*}} : tensor<4x1x8xf32> -> tensor<4x16x8xf32>
  %0 = tt.broadcast %arg0 : tensor<1x4x1x8xf32> -> tensor<1x4x16x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 0 : i32} : tensor<1x4x16x8xf32> -> tensor<4x16x8xf32>
  tt.return %1 : tensor<4x16x8xf32>
}

// CHECK-LABEL: func @push_squeeze_dims_up_through_trans
tt.func @push_squeeze_dims_up_through_trans(%arg0: tensor<4x1x8xf32>) -> tensor<8x4xf32> {
  // CHECK: tt.trans {{.*}} {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  %0 = tt.trans %arg0 {order = array<i32: 2, 0, 1>} : tensor<4x1x8xf32> -> tensor<8x4x1xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 2 : i32} : tensor<8x4x1xf32> -> tensor<8x4xf32>
  tt.return %1 : tensor<8x4xf32>
}

// CHECK-LABEL: func @push_squeeze_dims_up_through_join
tt.func @push_squeeze_dims_up_through_join(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<4x2xf32> {
  // CHECK-DAG: tt.join {{.*}} : tensor<4xf32> -> tensor<4x2xf32>
  %0 = tt.join %arg0, %arg1 : tensor<1x4xf32> -> tensor<1x4x2xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 0 : i32} : tensor<1x4x2xf32> -> tensor<4x2xf32>
  tt.return %1 : tensor<4x2xf32>
}

// CHECK-LABEL: func @push_squeeze_dims_up_through_reduce
tt.func @push_squeeze_dims_up_through_reduce(%arg0: tensor<8x4x1xf32>) -> tensor<8xf32> {
  // CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}> ({
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  // CHECK: }) : (tensor<8x4xf32>) -> tensor<8xf32>
  }) : (tensor<8x4x1xf32>) -> tensor<8x1xf32>
  %2 = triton_xla.squeeze_dims %0 {axis = 1 : i32} : tensor<8x1xf32> -> tensor<8xf32>
  tt.return %2 : tensor<8xf32>
}

// CHECK-LABEL: func @fold_squeeze_of_expand_cancelling
tt.func @fold_squeeze_of_expand_cancelling(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NOT: tt.expand_dims
  // CHECK-NOT: triton_xla.squeeze_dims
  %0 = tt.expand_dims %arg0 {axis = 1 : i32} : tensor<4x8xf32> -> tensor<4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @fold_squeeze_of_expand_swapping
tt.func @fold_squeeze_of_expand_swapping(%arg0: tensor<4x1x8xf32>) -> tensor<1x4x8xf32> {
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  // CHECK: tt.expand_dims {{.*}} {axis = 0 : i32} : tensor<4x8xf32> -> tensor<1x4x8xf32>
  %0 = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<4x1x8xf32> -> tensor<1x4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 2 : i32} : tensor<1x4x1x8xf32> -> tensor<1x4x8xf32>
  tt.return %1 : tensor<1x4x8xf32>
}

// CHECK-LABEL: func @squeeze_reshape
tt.func @squeeze_reshape(%arg0: tensor<4x1x1xf32>) -> tensor<4xf32> {
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 1 : i32} : tensor<4x1x1xf32> -> tensor<4x1xf32>
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 1 : i32} : tensor<4x1xf32> -> tensor<4xf32>
  %0 = tt.reshape %arg0 : tensor<4x1x1xf32> -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @expand_reshape
tt.func @expand_reshape(%arg0: tensor<4xf32>) -> tensor<4x1x1xf32> {
  %0 = tt.reshape %arg0 : tensor<4xf32> -> tensor<4x1x1xf32>
  // CHECK: tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
  // CHECK: tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<4x1xf32> -> tensor<4x1x1xf32>
  tt.return %0 : tensor<4x1x1xf32>
}

// CHECK-LABEL: func @skip_reshape_with_attr
tt.func @skip_reshape_with_attr(%arg0: tensor<4x1xf32>) -> tensor<4xf32> {
  // CHECK-NOT: triton_xla.squeeze_dims
  // CHECK: tt.reshape {{.*}} allow_reorder : tensor<4x1xf32> -> tensor<4xf32>
  %0 = tt.reshape %arg0 allow_reorder : tensor<4x1xf32> -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

#arg_enc = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#res_enc = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
// CHECK-LABEL: func @reshape_with_encoding
// CHECK-SAME:    tensor<1x32xf32, #[[ARG_ENC:.+]]>) -> tensor<32xf32, #[[RES_ENC:.+]]>
tt.func @reshape_with_encoding(%arg0: tensor<1x32xf32, #arg_enc>) -> tensor<32xf32, #res_enc> {
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 0 : i32} :
  // CHECK-SAME: tensor<1x32xf32, #[[ARG_ENC]]> -> tensor<32xf32, #[[RES_ENC]]>
  %0 = tt.reshape %arg0 : tensor<1x32xf32, #arg_enc> -> tensor<32xf32, #res_enc>
  tt.return %0 : tensor<32xf32, #res_enc>
}
}

// CHECK-LABEL: func @fold_squeeze_dims_of_load_ptr
tt.func @fold_squeeze_dims_of_load_ptr(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<4x8xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c128_i64 = arith.constant 128 : i64
  // CHECK: %[[ADDPTR:.*]] = tt.addptr %arg0, %c24_i64
  // CHECK: tt.make_tensor_ptr %[[ADDPTR]], {{.*}} {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
  %0 = tt.make_tensor_ptr %arg0, [%c4_i64, %c16_i64, %c8_i64], [%c128_i64, %c8_i64, %c1_i64], [%c0_i32, %c3_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<4x1x8xf32>>
  // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<4x8xf32>>
  %1 = tt.load %0 {boundaryCheck = array<i32: 2>, padding = 1 : i32} : !tt.ptr<tensor<4x1x8xf32>>
  // CHECK-NOT: triton_xla.squeeze_dims
  %2 = triton_xla.squeeze_dims %1 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %2 : tensor<4x8xf32>
}

// CHECK-LABEL: func @squeeze_dims_of_load_ptr_with_boundary_check
tt.func @squeeze_dims_of_load_ptr_with_boundary_check(%arg0: !tt.ptr<f32>) -> tensor<4x8xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %0 = tt.make_tensor_ptr %arg0, [%c4_i64, %c1_i64, %c8_i64], [%c8_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<4x1x8xf32>>
  // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<4x1x8xf32>>
  %1 = tt.load %0 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<4x1x8xf32>>
  // CHECK: triton_xla.squeeze_dims
  %2 = triton_xla.squeeze_dims %1 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %2 : tensor<4x8xf32>
}

// CHECK-LABEL: func @squeeze_dims_of_load_ptr_with_mask
tt.func @squeeze_dims_of_load_ptr_with_mask(%arg0: !tt.ptr<f32>, %arg1: tensor<4x1x8xi1>) -> tensor<4x8xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %0 = tt.make_tensor_ptr %arg0, [%c4_i64, %c1_i64, %c8_i64], [%c8_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<4x1x8xf32>>
  // CHECK: tt.load {{.*}}, %arg1 : !tt.ptr<tensor<4x1x8xf32>>
  %1 = tt.load %0, %arg1 : !tt.ptr<tensor<4x1x8xf32>>
  // CHECK: triton_xla.squeeze_dims
  %2 = triton_xla.squeeze_dims %1 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %2 : tensor<4x8xf32>
}

// CHECK-LABEL: func @squeeze_store
tt.func @squeeze_store(%arg0: !tt.ptr<f32>, %arg1: tensor<4x1x8xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %c16_i64 = arith.constant 16 : i64
  %c128_i64 = arith.constant 128 : i64
  // CHECK-DAG: triton_xla.squeeze_dims {{.*}} {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  // CHECK-DAG: %[[ADDPTR:.*]] = tt.addptr %arg0, %c24_i64
  // CHECK-DAG: tt.make_tensor_ptr %[[ADDPTR]], {{.*}} {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
  %0 = tt.make_tensor_ptr %arg0, [%c4_i64, %c16_i64, %c8_i64], [%c128_i64, %c8_i64, %c1_i64], [%c0_i32, %c3_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<4x1x8xf32>>
  // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<4x8xf32>>
  tt.store %0, %arg1 {boundaryCheck = array<i32: 2>} : !tt.ptr<tensor<4x1x8xf32>>
  tt.return
}

// CHECK-LABEL: func @squeeze_store_unit_tensor
tt.func @squeeze_store_unit_tensor(%arg0: !tt.ptr<f32>, %arg1: tensor<1x1xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.make_tensor_ptr %arg0, [%c1_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0>} : <tensor<1x1xf32>>
  // CHECK: triton_xla.squeeze_dims
  // CHECK: tt.store {{.*}} : !tt.ptr<tensor<1xf32>>
  tt.store %0, %arg1 : !tt.ptr<tensor<1x1xf32>>
  tt.return
}

// CHECK-LABEL: func @squeeze_store_with_mask
tt.func @squeeze_store_with_mask(%arg0: !tt.ptr<f32>, %arg1: tensor<4x1xf32>, %arg2: tensor<4x1xi1>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %0 = tt.make_tensor_ptr %arg0, [%c4_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0>} : <tensor<4x1xf32>>
  // CHECK-NOT: triton_xla.squeeze_dims
  // CHECK: tt.store {{.*}} : !tt.ptr<tensor<4x1xf32>>
  tt.store %0, %arg1, %arg2 : !tt.ptr<tensor<4x1xf32>>
  tt.return
}

// CHECK-LABEL: func @reorder_squeeze_dims
tt.func @reorder_squeeze_dims(%arg0: tensor<4x1x8x1xf32>) -> tensor<4x8xf32> {
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 1 : i32} : tensor<4x1x8x1xf32> -> tensor<4x8x1xf32>
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 2 : i32} : tensor<4x8x1xf32> -> tensor<4x8xf32>
  %0 = triton_xla.squeeze_dims %arg0 {axis = 3 : i32} : tensor<4x1x8x1xf32> -> tensor<4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @diamond
tt.func @diamond(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NOT: arith.negf {{.*}} : tensor<4x1x8xf32>
  %0 = arith.negf %arg0 : tensor<4x1x8xf32>
  %1 = arith.addf %0, %0 : tensor<4x1x8xf32>
  %2 = triton_xla.squeeze_dims %1 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %2 : tensor<4x8xf32>
}

// CHECK-LABEL: func @insert_expand_dims
tt.func @insert_expand_dims(%arg0: tensor<4x1xf32>) -> tensor<4xf32> {
  // CHECK: %[[NEGF:.*]] = arith.negf {{.*}} : tensor<4xf32>
  // CHECK-NOT: arith.negf
  // CHECK: tt.expand_dims %[[NEGF]] {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
  %0 = arith.negf %arg0 : tensor<4x1xf32>
  %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %2 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %2 : f32
  }) : (tensor<4x1xf32>) -> tensor<4xf32>
  %3 = tt.expand_dims %1 {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
  %4 = arith.addf %0, %3 : tensor<4x1xf32>
  %5 = triton_xla.squeeze_dims %4 {axis = 1 : i32} : tensor<4x1xf32> -> tensor<4xf32>
  tt.return %5 : tensor<4xf32>
}

// FINALIZE-LABEL: func @squeeze_dims_to_reshape
tt.func @squeeze_dims_to_reshape(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // FINALIZE: tt.reshape {{.*}} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  %0 = triton_xla.squeeze_dims %arg0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %0 : tensor<4x8xf32>
}
