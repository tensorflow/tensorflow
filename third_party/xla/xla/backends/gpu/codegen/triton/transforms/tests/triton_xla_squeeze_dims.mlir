// RUN: xla-opt %s --triton-xla-squeeze-dims="finalize=false" \
// RUN: --split-input-file | FileCheck %s

// RUN: xla-opt %s --triton-xla-squeeze-dims --split-input-file \
// RUN: | FileCheck %s --check-prefix=FINALIZE

// CHECK-LABEL: func @push_squeeze_dims_up_through_elementwise
func.func @push_squeeze_dims_up_through_elementwise(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // CHECK: arith.negf {{.*}} : tensor<4x8xf32>
  %0 = arith.negf %arg0 : tensor<4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @push_squeeze_dims_up_through_multiple_results
func.func @push_squeeze_dims_up_through_multiple_results(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // CHECK: tt.elementwise_inline_asm {{.*}} : tensor<4x8xf32> -> tensor<4x8xf32>, tensor<4x8xf32>
  %0:2 = tt.elementwise_inline_asm "" {constraints = "", packed_element = 1 : i32, pure = true} %arg0 : tensor<4x1x8xf32> -> tensor<4x1x8xf32>, tensor<4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0#0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @push_squeeze_dims_up_through_broadcast
func.func @push_squeeze_dims_up_through_broadcast(%arg0: tensor<1x4x1x8xf32>) -> tensor<4x16x8xf32> {
  // CHECK: tt.broadcast {{.*}} : tensor<4x1x8xf32> -> tensor<4x16x8xf32>
  %0 = tt.broadcast %arg0 : tensor<1x4x1x8xf32> -> tensor<1x4x16x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 0 : i32} : tensor<1x4x16x8xf32> -> tensor<4x16x8xf32>
  return %1 : tensor<4x16x8xf32>
}

// -----

// CHECK-LABEL: func @push_squeeze_dims_up_through_trans
func.func @push_squeeze_dims_up_through_trans(%arg0: tensor<4x1x8xf32>) -> tensor<8x4xf32> {
  // CHECK: tt.trans {{.*}} {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  %0 = tt.trans %arg0 {order = array<i32: 2, 0, 1>} : tensor<4x1x8xf32> -> tensor<8x4x1xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 2 : i32} : tensor<8x4x1xf32> -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// -----

// CHECK-LABEL: func @push_squeeze_dims_up_through_join
func.func @push_squeeze_dims_up_through_join(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<4x2xf32> {
  // CHECK-DAG: tt.join {{.*}} : tensor<4xf32> -> tensor<4x2xf32>
  %0 = tt.join %arg0, %arg1 : tensor<1x4xf32> -> tensor<1x4x2xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 0 : i32} : tensor<1x4x2xf32> -> tensor<4x2xf32>
  return %1 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: func @push_squeeze_dims_up_through_reduce
func.func @push_squeeze_dims_up_through_reduce(%arg0: tensor<8x4x1xf32>) -> tensor<8xf32> {
  // CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}> ({
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  // CHECK: }) : (tensor<8x4xf32>) -> tensor<8xf32>
  }) : (tensor<8x4x1xf32>) -> tensor<8x1xf32>
  %2 = triton_xla.squeeze_dims %0 {axis = 1 : i32} : tensor<8x1xf32> -> tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: func @fold_squeeze_of_expand_cancelling
func.func @fold_squeeze_of_expand_cancelling(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NOT: tt.expand_dims
  // CHECK-NOT: triton_xla.squeeze_dims
  %0 = tt.expand_dims %arg0 {axis = 1 : i32} : tensor<4x8xf32> -> tensor<4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @fold_squeeze_of_expand_swapping
func.func @fold_squeeze_of_expand_swapping(%arg0: tensor<4x1x8xf32>) -> tensor<1x4x8xf32> {
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  // CHECK: tt.expand_dims {{.*}} {axis = 0 : i32} : tensor<4x8xf32> -> tensor<1x4x8xf32>
  %0 = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<4x1x8xf32> -> tensor<1x4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 2 : i32} : tensor<1x4x1x8xf32> -> tensor<1x4x8xf32>
  return %1 : tensor<1x4x8xf32>
}

// -----

// CHECK-LABEL: func @squeeze_reshape
func.func @squeeze_reshape(%arg0: tensor<4x1x1xf32>) -> tensor<4xf32> {
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 1 : i32} : tensor<4x1x1xf32> -> tensor<4x1xf32>
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 1 : i32} : tensor<4x1xf32> -> tensor<4xf32>
  %0 = tt.reshape %arg0 : tensor<4x1x1xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @expand_reshape
func.func @expand_reshape(%arg0: tensor<4xf32>) -> tensor<4x1x1xf32> {
  %0 = tt.reshape %arg0 : tensor<4xf32> -> tensor<4x1x1xf32>
  // CHECK: tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<4xf32> -> tensor<4x1xf32>
  // CHECK: tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<4x1xf32> -> tensor<4x1x1xf32>
  return %0 : tensor<4x1x1xf32>
}

// -----

// CHECK-LABEL: func @skip_reshape_with_attr
func.func @skip_reshape_with_attr(%arg0: tensor<4x1xf32>) -> tensor<4xf32> {
  // CHECK-NOT: triton_xla.squeeze_dims
  // CHECK: tt.reshape {{.*}} allow_reorder : tensor<4x1xf32> -> tensor<4xf32>
  %0 = tt.reshape %arg0 allow_reorder : tensor<4x1xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

#arg_enc = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#res_enc = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
// CHECK-LABEL: func @reshape_with_encoding
// CHECK-SAME:    tensor<1x32xf32, #[[ARG_ENC:.+]]>) -> tensor<32xf32, #[[RES_ENC:.+]]>
func.func @reshape_with_encoding(%arg0: tensor<1x32xf32, #arg_enc>) -> tensor<32xf32, #res_enc> {
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 0 : i32} :
  // CHECK-SAME: tensor<1x32xf32, #[[ARG_ENC]]> -> tensor<32xf32, #[[RES_ENC]]>
  %0 = tt.reshape %arg0 : tensor<1x32xf32, #arg_enc> -> tensor<32xf32, #res_enc>
  return %0 : tensor<32xf32, #res_enc>
}
}

// -----

// CHECK-LABEL: func @fold_squeeze_dims_of_extract
// CHECK-SAME: (%[[INPUT:.*]]: memref<4x16x8xf32, #xtile.layout<[2, 1, 0]>>,
func.func @fold_squeeze_dims_of_extract(
  %input: memref<4x16x8xf32, #xtile.layout<[2, 1, 0]>>, %offset: index)  -> tensor<4x8xf32>
{
  // CHECK: %[[EXTRACT:.*]] = xtile.extract %[[INPUT]]
  // CHECK-SAME: memref<4x16x8xf32, #xtile.layout<[2, 1, 0]>> -> tensor<4x8xf32>
  %tile = xtile.extract %input[%offset, %offset, %offset][4, 1, 8][1, 1, 1]
    : memref<4x16x8xf32, #xtile.layout<[2, 1, 0]>> -> tensor<4x1x8xf32>
  // CHECK-NOT: triton_xla.squeeze_dims
  %squeezed = triton_xla.squeeze_dims %tile {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  // CHECK: return %[[EXTRACT]]
  return %squeezed : tensor<4x8xf32>
}


// -----

// CHECK-LABEL: func @squeeze_insert(
// CHECK-SAME: %[[BUFFER:.*]]: memref<4x16x8xf32, #xtile.layout<[2, 1, 0]>>,
// CHECK-SAME: %[[TILE:.*]]: tensor<4x1x8xf32>
func.func @squeeze_insert(
  %arg0: memref<4x16x8xf32, #xtile.layout<[2, 1, 0]>>,
  %arg1: tensor<4x1x8xf32>,
  %offset: index) {
  // CHECK: %[[REDUCED:.*]] = triton_xla.squeeze_dims %[[TILE]]
  // CHECK-SAME: {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  // CHECK: xtile.insert %[[REDUCED]] into %[[BUFFER]]
  // CHECK-SAME: tensor<4x8xf32> -> memref<4x16x8xf32, #xtile.layout<[2, 1, 0]>>
  xtile.insert %arg1 into %arg0[%offset, %offset, %offset][4, 1, 8][1, 1, 1]
    : tensor<4x1x8xf32> -> memref<4x16x8xf32, #xtile.layout<[2, 1, 0]>>
  return
}

// -----

// CHECK-LABEL: func @squeeze_insert_unit_tensor
func.func @squeeze_insert_unit_tensor(
  %arg0: memref<1x1xf32,#xtile.layout<[0, 1]>>,
  %arg1: tensor<1x1xf32>,
  %offset: index) {
  // CHECK: triton_xla.squeeze_dims
  // CHECK: xtile.insert {{.*}} : tensor<1xf32>
  xtile.insert %arg1 into %arg0[%offset, %offset] [1, 1] [1, 1]
    : tensor<1x1xf32> -> memref<1x1xf32,#xtile.layout<[0, 1]>>
  return
}

// -----

// CHECK-LABEL: func @reorder_squeeze_dims
func.func @reorder_squeeze_dims(%arg0: tensor<4x1x8x1xf32>) -> tensor<4x8xf32> {
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 1 : i32} : tensor<4x1x8x1xf32> -> tensor<4x8x1xf32>
  // CHECK: triton_xla.squeeze_dims {{.*}} {axis = 2 : i32} : tensor<4x8x1xf32> -> tensor<4x8xf32>
  %0 = triton_xla.squeeze_dims %arg0 {axis = 3 : i32} : tensor<4x1x8x1xf32> -> tensor<4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @diamond
func.func @diamond(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NOT: arith.negf {{.*}} : tensor<4x1x8xf32>
  %0 = arith.negf %arg0 : tensor<4x1x8xf32>
  %1 = arith.addf %0, %0 : tensor<4x1x8xf32>
  %2 = triton_xla.squeeze_dims %1 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @insert_expand_dims
func.func @insert_expand_dims(%arg0: tensor<4x1xf32>) -> tensor<4xf32> {
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
  return %5 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @push_squeeze_dims_up_through_if
func.func @push_squeeze_dims_up_through_if(
    %arg0: tensor<16xf32>, %arg1: tensor<4x1xf32>, %cond: i1
) -> (tensor<16xf32>, tensor<4xf32>) {
  // CHECK: scf.if %{{.*}} -> (tensor<16xf32>, tensor<4xf32>) {
  %if:2 = scf.if %cond -> (tensor<16xf32>, tensor<4x1xf32>) {
    // CHECK: %[[SQUEEZE:.*]] = triton_xla.squeeze_dims
    // CHECK: scf.yield %arg0, %[[SQUEEZE]] : tensor<16xf32>, tensor<4xf32>
    scf.yield %arg0, %arg1 : tensor<16xf32>, tensor<4x1xf32>
  } else {
    // CHECK: %[[SQUEEZE:.*]] = triton_xla.squeeze_dims
    // CHECK: scf.yield %arg0, %[[SQUEEZE]] : tensor<16xf32>, tensor<4xf32>
    scf.yield %arg0, %arg1 : tensor<16xf32>, tensor<4x1xf32>
  }
  // CHECK-NOT: triton_xla.squeeze_dims
  %squeeze = triton_xla.squeeze_dims %if#1 {axis = 1 : i32}
      : tensor<4x1xf32> -> tensor<4xf32>
  // CHECK: return
  return %if#0, %squeeze : tensor<16xf32>, tensor<4xf32>
}

// -----

// FINALIZE-LABEL: func @squeeze_dims_to_reshape
func.func @squeeze_dims_to_reshape(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // FINALIZE: tt.reshape {{.*}} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  %0 = triton_xla.squeeze_dims %arg0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @push_squeeze_dims_up_through_mask
func.func @push_squeeze_dims_up_through_mask(
    %arg0: tensor<4x1x8xf32>, %arg1: f32) -> tensor<4x8xf32> {
  // CHECK: xtile.mask %{{.*}} bounds [4, 6], %arg1 : tensor<4x8xf32>
  %0 = xtile.mask %arg0 bounds [4, 1, 6], %arg1 : tensor<4x1x8xf32>
  %1 = triton_xla.squeeze_dims %0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}
