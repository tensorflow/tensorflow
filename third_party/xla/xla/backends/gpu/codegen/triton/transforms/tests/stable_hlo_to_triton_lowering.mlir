// RUN: xla-opt %s -split-input-file \
// RUN: -stablehlo-lower-to-triton="warp_specialization_allowed=false" \
// RUN: | FileCheck %s
// RUN: xla-opt %s -split-input-file \
// RUN: -stablehlo-lower-to-triton="warp_specialization_allowed=true" \
// RUN: | FileCheck %s --check-prefix=WARP

// CHECK: func @lower_transpose(%[[ARG:.*]]: tensor<2x4x8xf32>) -> tensor<8x2x4xf32>
func.func @lower_transpose(%arg0: tensor<2x4x8xf32>) -> tensor<8x2x4xf32> {
  // CHECK: %[[RES:.*]] = tt.trans %[[ARG]] {order = array<i32: 2, 0, 1>} : tensor<2x4x8xf32> -> tensor<8x2x4xf32>
  %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<2x4x8xf32>) -> tensor<8x2x4xf32>
  // CHECK: return %[[RES]] : tensor<8x2x4xf32>
  return %0 : tensor<8x2x4xf32>
}

// CHECK: func @lower_iota_to_make_range() -> tensor<16xi32>
func.func @lower_iota_to_make_range() -> tensor<16xi32> {
  // CHECK: %[[RES:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %0 = stablehlo.iota dim = 0 : tensor<16xi32>
  // CHECK: return %[[RES]] : tensor<16xi32>
  return %0 : tensor<16xi32>
}

// CHECK: func @lower_iota_on_multidimensional_tensor_falls_back_to_stablehlo() -> tensor<16x32xi32>
func.func @lower_iota_on_multidimensional_tensor_falls_back_to_stablehlo() -> tensor<16x32xi32> {
  // CHECK: %[[RES:.*]] = stablehlo.iota dim = 0 : tensor<16x32xi32>
  %0 = stablehlo.iota dim = 0 : tensor<16x32xi32>
  // CHECK: return %[[RES]] : tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// CHECK: func @lower_iota_on_non_signed_32_bit_tensor_falls_back_to_stablehlo() -> tensor<8xui32>
func.func @lower_iota_on_non_signed_32_bit_tensor_falls_back_to_stablehlo() -> tensor<8xui32> {
  // CHECK: %[[RES:.*]] = stablehlo.iota dim = 0 : tensor<8xui32>
  %0 = stablehlo.iota dim = 0 : tensor<8xui32>
  // CHECK: return %[[RES]] : tensor<8xui32>
  return %0 : tensor<8xui32>
}

// CHECK: func @lower_broadcast_in_dim(%[[ARG0:.*]]: tensor<2x4xf32>) -> tensor<8x2x4x16xf32>
func.func @lower_broadcast_in_dim(%arg0: tensor<2x4xf32>) -> tensor<8x2x4x16xf32> {
  // CHECK: %[[RES_EXPAND_DIMS_0:.*]] = tt.expand_dims %[[ARG0]] {axis = 0 : i32} : tensor<2x4xf32> -> tensor<1x2x4xf32>
  // CHECK: %[[RES_EXPAND_DIMS_1:.*]] = tt.expand_dims %[[RES_EXPAND_DIMS_0]] {axis = 3 : i32} : tensor<1x2x4xf32> -> tensor<1x2x4x1xf32>
  // CHECK: %[[RES:.*]] = tt.broadcast %[[RES_EXPAND_DIMS_1]] : tensor<1x2x4x1xf32> -> tensor<8x2x4x16xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<2x4xf32>) -> tensor<8x2x4x16xf32>
  // CHECK: return %[[RES]] : tensor<8x2x4x16xf32>
  return %0 : tensor<8x2x4x16xf32>
}

// CHECK: func @lower_broadcast_in_dim_on_0d_tensor_produced_by_to_tensor_to_splat(%[[ARG0:.*]]: f32) -> tensor<4x2xf32>
func.func @lower_broadcast_in_dim_on_0d_tensor_produced_by_to_tensor_to_splat(%arg0: f32) -> tensor<4x2xf32> {
  // CHECK-NOT: tensor.from_elements
  // CHECK: %[[RES:.*]] = tt.splat %[[ARG0]] : f32 -> tensor<4x2xf32>
  %to_tensor = tensor.from_elements %arg0 : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %to_tensor, dims = [] : (tensor<f32>) -> tensor<4x2xf32>
  // CHECK: return %[[RES]] : tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK: func @reduce(%[[ARG0:.*]]: tensor<16x8xf32>) -> tensor<8xf32>
func.func @reduce(%arg0: tensor<16x8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RES:.*]] = "tt.reduce"(%[[ARG0]]) <{axis = 0 : i32}> ({
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  //CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    // CHECK: %[[ARG1_CAST:.*]] = tensor.from_elements %[[ARG1]] : tensor<f32>
    // CHECK: %[[ARG2_CAST:.*]] = tensor.from_elements %[[ARG2]] : tensor<f32>
    // CHECK: %[[RES:.*]] = arith.addf %[[ARG1_CAST]], %[[ARG2_CAST]] : tensor<f32>
    // CHECK: %[[RES_CAST:.*]] = tensor.extract %[[RES]][] : tensor<f32>
    // CHECK: tt.reduce.return %[[RES_CAST]] : f32
    %add = arith.addf %arg1, %arg2 : tensor<f32>
    stablehlo.return %add : tensor<f32>
  }) {dimensions = array<i64: 0>} : (tensor<16x8xf32>, tensor<f32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK: func @reduce_to_scalar_followed_by_extract(%[[ARG0:.*]]: tensor<16xf32>) -> f32
func.func @reduce_to_scalar_followed_by_extract(%arg0: tensor<16xf32>) -> f32 {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCE_RESULT:.*]] = "tt.reduce"(%[[ARG0]]) <{axis = 0 : i32}> ({
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  //CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    // CHECK: %[[RES:.*]] = arith.addf {{.*}} : tensor<f32>
    // CHECK: tt.reduce.return {{.*}} : f32
    %add = arith.addf %arg1, %arg2 : tensor<f32>
    stablehlo.return %add : tensor<f32>
  }) {dimensions = array<i64: 0>} : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
  // CHECK-NOT: tensor.from_elements
  // CHECK-NOT: tensor.extract
  %extract = tensor.extract %1[] : tensor<f32>
  // CHECK: return %[[REDUCE_RESULT:.*]] : f32
  return %extract : f32
}

// CHECK: func @reduce_over_multiple_dimensions_falls_back_to_stablehlo(%[[ARG0:.*]]: tensor<16x8x4xf32>) -> tensor<4xf32>
func.func @reduce_over_multiple_dimensions_falls_back_to_stablehlo(%arg0: tensor<16x8x4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.reduce(%[[ARG0]] init: %{{.*}}) across dimensions = [0, 1] : (tensor<16x8x4xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %add = arith.addf %arg1, %arg2 : tensor<f32>
    stablehlo.return %add : tensor<f32>
  }) {dimensions = array<i64: 0, 1>} : (tensor<16x8x4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: return %[[RES]] : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK: func @reduce_with_multiple_inputs(%[[ARG0:.*]]: tensor<16x8xf32>, %[[ARG1:.*]]: tensor<16x8xf32>) -> tensor<8xf32>
func.func @reduce_with_multiple_inputs(%arg0: tensor<16x8xf32>, %arg1: tensor<16x8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCE_RESULT:.*]] = "tt.reduce"(%[[ARG0]], %[[ARG1]]) <{axis = 0 : i32}> ({
  %1, %2 = "stablehlo.reduce"(%arg0, %arg1, %0, %0) ({
  ^bb0(%arg0_reducer: tensor<f32>, %arg1_reducer: tensor<f32>, %arg2_reducer: tensor<f32>, %arg3_reducer: tensor<f32>):
    %add0 = arith.addf %arg0_reducer, %arg1_reducer : tensor<f32>
    %add1 = arith.addf %arg2_reducer, %arg3_reducer : tensor<f32>
    stablehlo.return %add0, %add1 : tensor<f32>, tensor<f32>
  }) {dimensions = array<i64: 0>} : (tensor<16x8xf32>, tensor<16x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %1 : tensor<8xf32>
}

func.func @lower_reshape(%arg0: tensor<2x4x8xf32>) -> tensor<8x2x4xf32> {
  // CHECK: %[[RES:.*]] = tt.reshape %[[ARG]] : tensor<2x4x8xf32> -> tensor<8x2x4xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<2x4x8xf32>) -> tensor<8x2x4xf32>
  return %0 : tensor<8x2x4xf32>
}

// CHECK-LABEL: @reshape_0d_to_0d_folds(%arg0: tensor<f32>)
func.func @reshape_0d_to_0d_folds(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.reshape %arg0 : (tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @reshape_0d_to_2d_splats(%arg0: tensor<f32>)
func.func @reshape_0d_to_2d_splats(%arg0: tensor<f32>) -> tensor<1x1xf32> {
  // CHECK: %[[SCALAR:.*]] = tensor.extract %arg0[] : tensor<f32>
  // CHECK: %[[SPLAT:.*]] = tt.splat %[[SCALAR]] : f32 -> tensor<1x1xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<f32>) -> tensor<1x1xf32>
  // CHECK: return %[[SPLAT]]
  return %0 : tensor<1x1xf32>
}

// CHECK-LABEL: @reshape_2d_to_0d_reduces(%arg0: tensor<1x1xf32>)
func.func @reshape_2d_to_0d_reduces(%arg0: tensor<1x1xf32>) -> tensor<f32> {
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %arg0 allow_reorder : tensor<1x1xf32> -> tensor<1xf32>
  // CHECK: %[[REDUCE:.*]] = "tt.reduce"(%[[RESHAPE]]) <{axis = 0 : i32}> ({
  // CHECK:  ^bb0(%arg1: f32, %arg2: f32):
  // CHECK:    %[[ADD:.*]] = arith.addf %arg1, %arg2 : f32
  // CHECK:    tt.reduce.return %[[ADD]] : f32
  // CHECK:  }) : (tensor<1xf32>) -> f32
  // CHECK:  %[[REDUCE_TENSOR:.*]] = tensor.from_elements %[[REDUCE]] : tensor<f32>
  %0 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<f32>
  // CHECK: return %[[REDUCE_TENSOR]]
  return %0 : tensor<f32>
}

// CHECK: func @lower_dot_add_to_triton(%[[ARG0:.*]]: tensor<2x4xf32>, %[[ARG1:.*]]: tensor<4x8xf32>, %[[ARG2:.*]]: tensor<2x8xf32>) -> tensor<2x8xf32>
func.func @lower_dot_add_to_triton(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // CHECK: %[[RES:.*]] = tt.dot %[[ARG0]], %[[ARG1]], %[[ARG2]], inputPrecision = tf32 : tensor<2x4xf32> * tensor<4x8xf32> -> tensor<2x8xf32>
  // CHECK-NOT: arith.addf
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
  %1 = arith.addf %0, %arg2 : tensor<2x8xf32>
  // CHECK: return %[[RES]] : tensor<2x8xf32>
  return %1 : tensor<2x8xf32>
}

// CHECK: func @lower_dot_without_add_falls_back_to_stablehlo(%[[ARG0:.*]]: tensor<2x4xf32>, %[[ARG1:.*]]: tensor<4x8xf32>, %[[ARG2:.*]]: tensor<2x8xf32>) -> tensor<2x8xf32>
func.func @lower_dot_without_add_falls_back_to_stablehlo(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // CHECK: %[[RES:.*]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK: return %[[RES]] : tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// CHECK: func @lower_dot_f8_no_ieee_has_max_num_imprecise_acc_set_to_max(%[[ARG0:.*]]: tensor<2x4xf8E4M3FN>, %[[ARG1:.*]]: tensor<4x8xf8E4M3FN>, %[[ARG2:.*]]: tensor<2x8xf8E4M3FN>) -> tensor<2x8xf8E4M3FN>
func.func @lower_dot_f8_no_ieee_has_max_num_imprecise_acc_set_to_max(%arg0: tensor<2x4xf8E4M3FN>, %arg1: tensor<4x8xf8E4M3FN>, %arg2: tensor<2x8xf8E4M3FN>) -> tensor<2x8xf8E4M3FN> {
  // CHECK: %[[RES:.*]] = tt.dot %[[ARG0]], %[[ARG1]], %[[ARG2]], inputPrecision = tf32 {maxNumImpreciseAcc = 2147483647 : i32} : tensor<2x4xf8E4M3FN> * tensor<4x8xf8E4M3FN> -> tensor<2x8xf8E4M3FN>
  // CHECK-NOT: arith.addf
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf8E4M3FN>, tensor<4x8xf8E4M3FN>) -> tensor<2x8xf8E4M3FN>
  %1 = arith.addf %0, %arg2 : tensor<2x8xf8E4M3FN>
  // CHECK: return %[[RES]] : tensor<2x8xf8E4M3FN>
  return %1 : tensor<2x8xf8E4M3FN>
}

func.func @all_reduce_without_xtile_entry_func_doesnt_lower(%input: tensor<10xf32>, %output: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: stablehlo.all_reduce
  %all_reduce = "stablehlo.all_reduce"(%input) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
    ^bb0(%arg7: tensor<f32>, %arg8: tensor<f32>):
      %4 = arith.addf %arg7, %arg8 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<10xf32>) -> tensor<10xf32>
  return %all_reduce : tensor<10xf32>
}

xtile.entry_func @all_reduce_with_multiple_inputs_doesnt_lower(%input: memref<1024xf32>, %output: memref<1024xf32>, %device_rank: i32, %signal_value: i32, %signal_buffer: !tt.ptr<!tt.ptr<i32>>, %remote_input_buffer: !tt.ptr<!tt.ptr<i64>>, %tile_id: index) attributes {num_opaque_args = 4 : i32} {
  %tile = xtile.extract %input[%tile_id][10][1] : memref<1024xf32> -> tensor<10xf32>
  %c_1 = arith.constant 1 : index
  %tile_id_2 = arith.addi %tile_id, %c_1 : index
  %tile2 = xtile.extract %input[%tile_id_2][10][1] : memref<1024xf32> -> tensor<10xf32>
  // CHECK: stablehlo.all_reduce
  %all_reduce:2 = "stablehlo.all_reduce"(%tile, %tile2) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
    ^bb0(%arg7: tensor<f32>, %arg8: tensor<f32>):
      %4 = arith.addf %arg7, %arg8 : tensor<f32>
      stablehlo.return %4: tensor<f32>
    }) : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)
  xtile.return
}

xtile.entry_func @all_reduce_with_multiple_operations_in_reducer_doesnt_lower(%input: memref<1024xf32>, %output: memref<1024xf32>, %device_rank: i32, %signal_value: i32, %signal_buffer: !tt.ptr<!tt.ptr<i32>>, %remote_input_buffer: !tt.ptr<!tt.ptr<i64>>, %tile_id: index) attributes {num_opaque_args = 4 : i32} {
  %tile = xtile.extract %input[%tile_id][10][1] : memref<1024xf32> -> tensor<10xf32>
  // CHECK: stablehlo.all_reduce
  %all_reduce = "stablehlo.all_reduce"(%tile) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
    ^bb0(%arg7: tensor<f32>, %arg8: tensor<f32>):
      %4 = arith.addf %arg7, %arg8 : tensor<f32>
      %5 = arith.addf %4, %arg8 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<10xf32>) -> tensor<10xf32>
  xtile.return
}

xtile.entry_func @all_reduce_input_not_from_extract_doesnt_lower(%input: memref<1024xf32>, %output: memref<1024xf32>, %device_rank: i32, %signal_value: i32, %signal_buffer: !tt.ptr<!tt.ptr<i32>>, %remote_input_buffer: !tt.ptr<!tt.ptr<i64>>, %tile_id: index) attributes {num_opaque_args = 4 : i32} {
  %tile = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
  // CHECK: stablehlo.all_reduce
  %all_reduce = "stablehlo.all_reduce"(%tile) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
    ^bb0(%arg7: tensor<f32>, %arg8: tensor<f32>):
      %4 = arith.addf %arg7, %arg8 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<10xf32>) -> tensor<10xf32>
  xtile.return
}

xtile.entry_func @all_reduce_with_incorrect_num_args_doesnt_lower(%input: memref<1024xf32>, %output: memref<1024xf32>, %device_rank: i32, %signal_value: i32, %signal_buffer: !tt.ptr<!tt.ptr<i32>>, %remote_input_buffer: !tt.ptr<!tt.ptr<i64>>, %dummy_arg: i32, %tile_id: index) attributes {num_opaque_args = 5 : i32} {
  %tile = xtile.extract %input[%tile_id][10][1] : memref<1024xf32> -> tensor<10xf32>
  // CHECK: stablehlo.all_reduce
  %all_reduce = "stablehlo.all_reduce"(%tile) <{replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
    ^bb0(%arg7: tensor<f32>, %arg8: tensor<f32>):
      %4 = arith.addf %arg7, %arg8 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<10xf32>) -> tensor<10xf32>
  xtile.return
}

// CHECK: func @lower_dot_with_warp_specialization_to_triton
func.func @lower_dot_with_warp_specialization_to_triton(
    %arg0: tensor<2x4xf32>,
    %arg1: tensor<4x8xf32>,
    %arg2: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %iv = %c0 to %c4 step %c1 iter_args(%accum = %arg2) -> tensor<2x8xf32> {
    %dot = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %add = arith.addf %dot, %accum : tensor<2x8xf32>
    // CHECK-NOT : tt.warp_specialize
    // WARP: scf.yield
    // WARP-NEXT: tt.warp_specialize = true
    scf.yield %add : tensor<2x8xf32>
  }
  return %res : tensor<2x8xf32>
}
