// RUN: litert-opt %s -tfl-rank-reduction | FileCheck %s

module {
  // CHECK-LABEL: lift_reshape_through_cast
  func.func @lift_reshape_through_cast(%arg0: tensor<1x25xi1>) -> tensor<25xf32> {
    %cst = "tfl.pseudo_const"() {value = dense<[25]> : tensor<1xi32>} : () -> tensor<1xi32>

    // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, %{{.*}}) : (tensor<1x25xi1>, tensor<1xi32>) -> tensor<25xi1>
    // CHECK: %[[CAST:.*]] = "tfl.cast"(%[[RESHAPE]]) : (tensor<25xi1>) -> tensor<25xf32>
    // CHECK: return %[[CAST]]
    %0 = "tfl.cast"(%arg0) : (tensor<1x25xi1>) -> tensor<1x25xf32>
    %1 = "tfl.reshape"(%0, %cst) : (tensor<1x25xf32>, tensor<1xi32>) -> tensor<25xf32>
    func.return %1 : tensor<25xf32>
  }

  // CHECK-LABEL: lift_reshape_through_concatenation
  func.func @lift_reshape_through_concatenation(%arg0: tensor<1x1x2x12x24xi1>, %arg1: tensor<1x1x2x12x24xi1>) -> tensor<1x2x12x48xi1> {
    // CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x2x12x24xi1>, %[[ARG1:.*]]: tensor<1x1x2x12x24xi1>)
    // CHECK: %[[RESHAPE_CONST:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 2, 12, 24]> : tensor<4xi32>}>
    // CHECK: %[[RESHAPE_0:.*]] = "tfl.reshape"(%[[ARG0]], %[[RESHAPE_CONST]])
    // CHECK: %[[RESHAPE_1:.*]] = "tfl.reshape"(%[[ARG1]], %[[RESHAPE_CONST]])
    // CHECK: %[[CONCAT:.*]] = "tfl.concatenation"(%[[RESHAPE_0]], %[[RESHAPE_1]]) <{axis = 3 : i32, fused_activation_function = "NONE"}>
    // CHECK: return %[[CONCAT]]
    %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 4 : i32, fused_activation_function = "NONE"} : (tensor<1x1x2x12x24xi1>, tensor<1x1x2x12x24xi1>) -> tensor<1x1x2x12x48xi1>
    %cst = "tfl.pseudo_const"() {value = dense<[1, 2, 12, 48]> : tensor<4xi32>} : () -> tensor<4xi32>
    %1 = "tfl.reshape"(%0, %cst) : (tensor<1x1x2x12x48xi1>, tensor<4xi32>) -> tensor<1x2x12x48xi1>
    func.return %1 : tensor<1x2x12x48xi1>
  }

  // CHECK-LABEL: testBMMRankReduction
  func.func @testBMMRankReduction(%arg0: tensor<1x2x8x48x24xf32>, %arg1: tensor<1x2x8x24x12xf32>) -> tensor<1x2x8x48x12xf32> {
    // CHECK: %[[LHS_RESHAPE:.*]] = "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x8x48x24xf32>, tensor<4xi32>) -> tensor<2x8x48x24xf32>
    // CHECK: %[[RHS_RESHAPE:.*]] = "tfl.reshape"(%arg1, {{.*}}) : (tensor<1x2x8x24x12xf32>, tensor<4xi32>) -> tensor<2x8x24x12xf32>
    // CHECK: %[[BMM:.*]] = "tfl.batch_matmul"(%[[LHS_RESHAPE]], %[[RHS_RESHAPE]]) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<2x8x48x24xf32>, tensor<2x8x24x12xf32>) -> tensor<2x8x48x12xf32>
    // CHECK: %[[RES:.*]] = "tfl.reshape"(%[[BMM]], {{.*}}) : (tensor<2x8x48x12xf32>, tensor<5xi32>) -> tensor<1x2x8x48x12xf32>
    // CHECK: return %[[RES]]
    %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<1x2x8x48x24xf32>, tensor<1x2x8x24x12xf32>) -> tensor<1x2x8x48x12xf32>
    func.return %0 : tensor<1x2x8x48x12xf32>
  }

  // CHECK-LABEL: testBMMRankReductionWithBroadcasting
  func.func @testBMMRankReductionWithBroadcasting(%arg0: tensor<1x1x8x48x24xf32>, %arg1: tensor<1x2x1x24x12xf32>) -> tensor<1x2x8x48x12xf32> {
    // CHECK: %[[LHS_RESHAPE:.*]] = "tfl.reshape"
    // CHECK: %[[RHS_RESHAPE:.*]] = "tfl.reshape"
    // CHECK: %[[BMM:.*]] = "tfl.batch_matmul"(%[[LHS_RESHAPE]], %[[RHS_RESHAPE]])
    // CHECK: %[[RES:.*]] = "tfl.reshape"(%[[BMM]], {{.*}}) : (tensor<2x8x48x12xf32>, tensor<5xi32>) -> tensor<1x2x8x48x12xf32>
    %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<1x1x8x48x24xf32>, tensor<1x2x1x24x12xf32>) -> tensor<1x2x8x48x12xf32>
    func.return %0 : tensor<1x2x8x48x12xf32>
  }
  // CHECK-LABEL: fuse_consecutive_reshapes
  func.func @fuse_consecutive_reshapes(%arg0: tensor<1x2x8x48x24xf32>) -> tensor<16x48x24xf32> {
    // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x8x48x24xf32>, tensor<3xi32>) -> tensor<16x48x24xf32>
    // CHECK-NOT: "tfl.reshape"
    // CHECK: return %[[RESHAPE]]
    %c1 = "tfl.pseudo_const"() {value = dense<[2, 8, 48, 24]> : tensor<4xi32>} : () -> tensor<4xi32>
    %0 = "tfl.reshape"(%arg0, %c1) : (tensor<1x2x8x48x24xf32>, tensor<4xi32>) -> tensor<2x8x48x24xf32>
    %c2 = "tfl.pseudo_const"() {value = dense<[16, 48, 24]> : tensor<3xi32>} : () -> tensor<3xi32>
    %1 = "tfl.reshape"(%0, %c2) : (tensor<2x8x48x24xf32>, tensor<3xi32>) -> tensor<16x48x24xf32>
    func.return %1 : tensor<16x48x24xf32>
  }

  // CHECK-LABEL: testBMMRankReductionWithFuse
  func.func @testBMMRankReductionWithFuse(%arg0: tensor<1x2x8x48x24xf32>, %arg1: tensor<1x2x8x24x12xf32>) -> tensor<16x48x12xf32> {
    // CHECK: %[[LHS_RES:.*]] = "tfl.reshape"(%arg0, {{.*}})
    // CHECK: %[[RHS_RES:.*]] = "tfl.reshape"(%arg1, {{.*}})
    // CHECK: %[[BMM:.*]] = "tfl.batch_matmul"(%[[LHS_RES]], %[[RHS_RES]])
    // CHECK: %[[FINAL_RES:.*]] = "tfl.reshape"(%[[BMM]], {{.*}}) : (tensor<2x8x48x12xf32>, tensor<3xi32>) -> tensor<16x48x12xf32>
    // CHECK-NOT: "tfl.reshape"
    // CHECK: return %[[FINAL_RES]]
    %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x2x8x48x24xf32>, tensor<1x2x8x24x12xf32>) -> tensor<1x2x8x48x12xf32>
    %shape = "tfl.pseudo_const"() {value = dense<[16, 48, 12]> : tensor<3xi32>} : () -> tensor<3xi32>
    %1 = "tfl.reshape"(%0, %shape) : (tensor<1x2x8x48x12xf32>, tensor<3xi32>) -> tensor<16x48x12xf32>
    func.return %1 : tensor<16x48x12xf32>
  }

  // CHECK-LABEL: testBMMRankReductionRepro
  func.func @testBMMRankReductionRepro(%arg0: tensor<1x2x8x48x24xf32>, %arg1: tensor<1x2x8x24x12xf32>) -> tensor<1x2x8x48x12xf32> {
    // CHECK: %[[LHS_RESHAPE:.*]] = "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x8x48x24xf32>, tensor<4xi32>) -> tensor<2x8x48x24xf32>
    // CHECK: %[[RHS_RESHAPE:.*]] = "tfl.reshape"(%arg1, {{.*}}) : (tensor<1x2x8x24x12xf32>, tensor<4xi32>) -> tensor<2x8x24x12xf32>
    // CHECK: %[[BMM:.*]] = "tfl.batch_matmul"(%[[LHS_RESHAPE]], %[[RHS_RESHAPE]])
    // CHECK: %[[RES:.*]] = "tfl.reshape"(%[[BMM]], {{.*}}) : (tensor<2x8x48x12xf32>, tensor<5xi32>) -> tensor<1x2x8x48x12xf32>
    // CHECK: return %[[RES]]
    %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<1x2x8x48x24xf32>, tensor<1x2x8x24x12xf32>) -> tensor<1x2x8x48x12xf32>
    func.return %0 : tensor<1x2x8x48x12xf32>
  }

  // CHECK-LABEL: lift_reshape_through_concat
  func.func @lift_reshape_through_concat(%arg0: tensor<2x1x1x12x8x48xf32>, %arg1: tensor<2x1x1x12x8x48xf32>) -> tensor<2x24x8x48xf32> {
    // CHECK-DAG: %[[RESHAPE_CONST:.*]] = "tfl.pseudo_const"() <{value = dense<[2, 12, 8, 48]> : tensor<4xi32>}>
    // CHECK: %[[RESHAPE_0:.*]] = "tfl.reshape"(%arg0, %[[RESHAPE_CONST]])
    // CHECK: %[[RESHAPE_1:.*]] = "tfl.reshape"(%arg1, %[[RESHAPE_CONST]])
    // CHECK: %[[CONCAT:.*]] = "tfl.concatenation"(%[[RESHAPE_0]], %[[RESHAPE_1]]) <{axis = 1 : i32, fused_activation_function = "NONE"}>
    // CHECK: return %[[CONCAT]]
    %8 = "tfl.pseudo_const"() {value = dense<[2, 24, 8, 48]> : tensor<4xi32>} : () -> tensor<4xi32>
    %2493 = "tfl.concatenation"(%arg0, %arg1) <{axis = 1 : i32, fused_activation_function = "NONE"}> : (tensor<2x1x1x12x8x48xf32>, tensor<2x1x1x12x8x48xf32>) -> tensor<2x2x1x12x8x48xf32>
    %2494 = "tfl.reshape"(%2493, %8) : (tensor<2x2x1x12x8x48xf32>, tensor<4xi32>) -> tensor<2x24x8x48xf32>
    func.return %2494 : tensor<2x24x8x48xf32>
  }

  func.func @squeeze_broadcast_to(%arg0: tensor<1x12xf32>, %arg1: tensor<5xi32>) -> tensor<1x12x32x1x32xf32> {
    // CHECK-DAG: %[[TARGET_SHAPE:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 12, 32, 1, 32]> : tensor<5xi32>}> : () -> tensor<5xi32>
    // CHECK-DAG: %[[SQUEEZED_SHAPE:.*]] = "tfl.pseudo_const"() <{value = dense<[12, 32, 1, 32]> : tensor<4xi64>}> : () -> tensor<4xi64>
    // CHECK-DAG: %[[SQUEEZED_INPUT_SHAPE:.*]] = "tfl.pseudo_const"() <{value = dense<[12, 1, 1, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    // CHECK: %[[SQUEEZED_INPUT:.*]] = "tfl.reshape"(%arg0, %[[SQUEEZED_INPUT_SHAPE]]) : (tensor<1x12xf32>, tensor<4xi32>) -> tensor<12x1x1x1xf32>
    // CHECK: %[[SQUEEZED_BROADCAST:.*]] = "tfl.broadcast_to"(%[[SQUEEZED_INPUT]], %[[SQUEEZED_SHAPE]]) : (tensor<12x1x1x1xf32>, tensor<4xi64>) -> tensor<12x32x1x32xf32>
    // CHECK: %[[OUT:.*]] = "tfl.reshape"(%[[SQUEEZED_BROADCAST]], %[[TARGET_SHAPE]]) : (tensor<12x32x1x32xf32>, tensor<5xi32>) -> tensor<1x12x32x1x32xf32>
    // CHECK: return %[[OUT]]
    %0 = "tfl.pseudo_const"() {value = dense<[1, 12, 32, 1, 32]> : tensor<5xi64>} : () -> tensor<5xi64>
    %1 = "tfl.reshape"(%arg0, %arg1) : (tensor<1x12xf32>, tensor<5xi32>) -> tensor<1x12x1x1x1xf32>
    %2 = "tfl.broadcast_to"(%1, %0) : (tensor<1x12x1x1x1xf32>, tensor<5xi64>) -> tensor<1x12x32x1x32xf32>
    func.return %2 : tensor<1x12x32x1x32xf32>
  }

  // CHECK-LABEL: reduce_mul_rank
  func.func @reduce_mul_rank(%arg0: tensor<1x25x64x128xf32>, %arg1: tensor<1x25xf32>) -> tensor<1x25x64x1x128xf32> {
    %cst_lhs = "tfl.pseudo_const"() {value = dense<[1, 25, 64, 1, 128]> : tensor<5xi32>} : () -> tensor<5xi32>
    %cst_rhs = "tfl.pseudo_const"() {value = dense<[1, 25, 1, 1, 1]> : tensor<5xi32>} : () -> tensor<5xi32>
    // CHECK-DAG: %[[OUT_CST:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 25, 64, 1, 128]> : tensor<5xi32>}>
    // CHECK-DAG: %[[RHS_CST:.*]] = "tfl.pseudo_const"() <{value = dense<[25, 1, 1, 1]> : tensor<4xi32>}>
    // CHECK-DAG: %[[LHS_CST:.*]] = "tfl.pseudo_const"() <{value = dense<[25, 64, 1, 128]> : tensor<4xi32>}>
    // CHECK: %[[RESHAPE_LHS:.*]] = "tfl.reshape"(%arg0, %[[LHS_CST]]) : (tensor<1x25x64x128xf32>, tensor<4xi32>) -> tensor<25x64x1x128xf32>
    // CHECK: %[[RESHAPE_RHS:.*]] = "tfl.reshape"(%arg1, %[[RHS_CST]]) : (tensor<1x25xf32>, tensor<4xi32>) -> tensor<25x1x1x1xf32>
    // CHECK: %[[MUL:.*]] = tfl.mul(%[[RESHAPE_LHS]], %[[RESHAPE_RHS]])
    // CHECK: %[[RET:.*]] = "tfl.reshape"(%[[MUL]], %[[OUT_CST]])
    // CHECK: return %[[RET]]

    %0 = "tfl.reshape"(%arg0, %cst_lhs) : (tensor<1x25x64x128xf32>, tensor<5xi32>) -> tensor<1x25x64x1x128xf32>
    %1 = "tfl.reshape"(%arg1, %cst_rhs) : (tensor<1x25xf32>, tensor<5xi32>) -> tensor<1x25x1x1x1xf32>
    %2 = tfl.mul(%0, %1) {fused_activation_function = "NONE"} : (tensor<1x25x64x1x128xf32>, tensor<1x25x1x1x1xf32>) -> tensor<1x25x64x1x128xf32>
    func.return %2 : tensor<1x25x64x1x128xf32>
  }

  // CHECK-LABEL: @reduce_sum_rank
  func.func @reduce_sum_rank(%arg0: tensor<25x64x1x128xf32>) -> tensor<1x25x1x1x1xf32> {
    // CHECK-NOT: tfl.reshape
    // CHECK: %[[AXES:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 3]> : tensor<2xi32>}> : () -> tensor<2xi32>
    // CHECK: %[[SUM:.*]] = "tfl.sum"(%arg0, %[[AXES]]) <{keep_dims = true}> : (tensor<25x64x1x128xf32>, tensor<2xi32>) -> tensor<25x1x1x1xf32>
    // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%[[SUM]], %{{.*}}) : (tensor<25x1x1x1xf32>, tensor<5xi32>) -> tensor<1x25x1x1x1xf32>
    // CHECK: return %[[RESHAPE]]
    %cst_axes = "tfl.pseudo_const"() {value = dense<[2, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_shape = "tfl.pseudo_const"() {value = dense<[1, 25, 64, 1, 128]> : tensor<5xi32>} : () -> tensor<5xi32>
    %0 = "tfl.reshape"(%arg0, %cst_shape) : (tensor<25x64x1x128xf32>, tensor<5xi32>) -> tensor<1x25x64x1x128xf32>
    %1 = "tfl.sum"(%0, %cst_axes) {keep_dims = true} : (tensor<1x25x64x1x128xf32>, tensor<2xi32>) -> tensor<1x25x1x1x1xf32>
    func.return %1 : tensor<1x25x1x1x1xf32>
  }

  // CHECK-LABEL: @reduce_cumsum_rank
  func.func @reduce_cumsum_rank(%arg0: tensor<25x1x1x1xf32>) -> tensor<1x25x1x1x1xf32> {
    // CHECK-NOT: tfl.reshape
    // CHECK: %[[AXIS:.*]] = "tfl.pseudo_const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    // CHECK: %[[CUMSUM:.*]] = "tfl.cumsum"(%arg0, %[[AXIS]]) <{exclusive = false, reverse = false}> : (tensor<25x1x1x1xf32>, tensor<i32>) -> tensor<25x1x1x1xf32>
    // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%[[CUMSUM]], %{{.*}}) : (tensor<25x1x1x1xf32>, tensor<5xi32>) -> tensor<1x25x1x1x1xf32>
    // CHECK: return %[[RESHAPE]]
    %cst_axis = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst_shape = "tfl.pseudo_const"() {value = dense<[1, 25, 1, 1, 1]> : tensor<5xi32>} : () -> tensor<5xi32>
    %0 = "tfl.reshape"(%arg0, %cst_shape) : (tensor<25x1x1x1xf32>, tensor<5xi32>) -> tensor<1x25x1x1x1xf32>
    %1 = "tfl.cumsum"(%0, %cst_axis) {exclusive = false, reverse = false} : (tensor<1x25x1x1x1xf32>, tensor<i32>) -> tensor<1x25x1x1x1xf32>
    func.return %1 : tensor<1x25x1x1x1xf32>
  }

  // CHECK-LABEL: @reduce_maximum_rank
  func.func @reduce_maximum_rank(%arg0: tensor<1x25x1x1x1xf32>, %arg1: tensor<f32>) -> tensor<1x25x1x1x1xf32> {
    // CHECK-DAG: %[[OUT_CST:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 25, 1, 1, 1]> : tensor<5xi32>}> : () -> tensor<5xi32>
    // CHECK-DAG: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[25, 1, 1, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    // CHECK: %[[LHS:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<1x25x1x1x1xf32>, tensor<4xi32>) -> tensor<25x1x1x1xf32>
    // CHECK: %[[OP:.*]] = "tfl.maximum"(%[[LHS]], %arg1) : (tensor<25x1x1x1xf32>, tensor<f32>) -> tensor<25x1x1x1xf32>
    // CHECK: %[[RES:.*]] = "tfl.reshape"(%[[OP]], %[[OUT_CST]]) : (tensor<25x1x1x1xf32>, tensor<5xi32>) -> tensor<1x25x1x1x1xf32>
    // CHECK: return %[[RES]]
    %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<1x25x1x1x1xf32>, tensor<f32>) -> tensor<1x25x1x1x1xf32>
    func.return %0 : tensor<1x25x1x1x1xf32>
  }
 
  // CHECK-LABEL: @reduce_minimum_rank
  func.func @reduce_minimum_rank(%arg0: tensor<1x25x1x1x1xf32>, %arg1: tensor<1x1x1x1x1xf32>) -> tensor<1x25x1x1x1xf32> {
    // CHECK-DAG: %[[OUT_CST:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 25, 1, 1, 1]> : tensor<5xi32>}> : () -> tensor<5xi32>
    // CHECK-DAG: %[[RHS_CST:.*]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<4xi32>}> : () -> tensor<4xi32>
    // CHECK-DAG: %[[LHS_CST:.*]] = "tfl.pseudo_const"() <{value = dense<[25, 1, 1, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    // CHECK: %[[LHS:.*]] = "tfl.reshape"(%arg0, %[[LHS_CST]]) : (tensor<1x25x1x1x1xf32>, tensor<4xi32>) -> tensor<25x1x1x1xf32>
    // CHECK: %[[RHS:.*]] = "tfl.reshape"(%arg1, %[[RHS_CST]]) : (tensor<1x1x1x1x1xf32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
    // CHECK: %[[OP:.*]] = "tfl.minimum"(%[[LHS]], %[[RHS]]) : (tensor<25x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<25x1x1x1xf32>
    // CHECK: %[[RES:.*]] = "tfl.reshape"(%[[OP]], %[[OUT_CST]]) : (tensor<25x1x1x1xf32>, tensor<5xi32>) -> tensor<1x25x1x1x1xf32>
    // CHECK: return %[[RES]]
    %0 = "tfl.minimum"(%arg0, %arg1) : (tensor<1x25x1x1x1xf32>, tensor<1x1x1x1x1xf32>) -> tensor<1x25x1x1x1xf32>
    func.return %0 : tensor<1x25x1x1x1xf32>
  }

  // CHECK-LABEL: lift_reshape_through_slice
  func.func @lift_reshape_through_slice(%arg0: tensor<1x1x1x128x128xf32>) -> tensor<1x1x64x128xf32> {
    %cst_s = "tfl.pseudo_const"() {value = dense<[0, 0, 0, 0, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %cst_0 = "tfl.pseudo_const"() {value = dense<[1, 1, 1, 64, 128]> : tensor<5xi32>} : () -> tensor<5xi32>
    %cst_1 = "tfl.pseudo_const"() {value = dense<[1, 1, 64, 128]> : tensor<4xi32>} : () -> tensor<4xi32>

    // CHECK: "tfl.reshape"
    // CHECK-SAME: (tensor<1x1x1x128x128xf32>, tensor<4xi32>) -> tensor<1x1x128x128xf32>
    // CHECK: "tfl.slice"
    // CHECK-SAME: (tensor<1x1x128x128xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x1x64x128xf32>
    %0 = "tfl.slice"(%arg0, %cst_s, %cst_0) : (tensor<1x1x1x128x128xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<1x1x1x64x128xf32>
    %1 = "tfl.reshape"(%0, %cst_1) : (tensor<1x1x1x64x128xf32>, tensor<4xi32>) -> tensor<1x1x64x128xf32>

    func.return %1 : tensor<1x1x64x128xf32>
  }

  // CHECK-LABEL: @reduce_slice_rank
  func.func @reduce_slice_rank(%arg0: tensor<1x25x1x1x1xf32>) -> tensor<1x1x1x1x1xf32> {
    // CHECK-DAG: %[[OUT_CST:.*]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<5xi32>}> : () -> tensor<5xi32>
    // CHECK-DAG: %[[BEGIN_CST:.*]] = "tfl.pseudo_const"() <{value = dense<[23, 0, 0, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
    // CHECK-DAG: %[[SIZE_CST:.*]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<4xi32>}> : () -> tensor<4xi32>
    // CHECK-DAG: %[[IN_RESHAPE_CST:.*]] = "tfl.pseudo_const"() <{value = dense<[25, 1, 1, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    // CHECK: %[[IN_RESHAPE:.*]] = "tfl.reshape"(%arg0, %[[IN_RESHAPE_CST]]) : (tensor<1x25x1x1x1xf32>, tensor<4xi32>) -> tensor<25x1x1x1xf32>
    // CHECK: %[[SLICE:.*]] = "tfl.slice"(%[[IN_RESHAPE]], %[[BEGIN_CST]], %[[SIZE_CST]]) : (tensor<25x1x1x1xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
    // CHECK: %[[OUT_RESHAPE:.*]] = "tfl.reshape"(%[[SLICE]], %[[OUT_CST]]) : (tensor<1x1x1x1xf32>, tensor<5xi32>) -> tensor<1x1x1x1x1xf32>
    // CHECK: return %[[OUT_RESHAPE]]
    %begin = "tfl.pseudo_const"() <{value = dense<[0, 23, 0, 0, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %size = "tfl.pseudo_const"() <{value = dense<1> : tensor<5xi32>}> : () -> tensor<5xi32>
    %0 = "tfl.slice"(%arg0, %begin, %size) : (tensor<1x25x1x1x1xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<1x1x1x1x1xf32>
    func.return %0 : tensor<1x1x1x1x1xf32>
  }

  func.func @repro(%arg0: tensor<2x1x12x8x48xf32>, %arg1: tensor<2x1x12x8x48xf32>) -> tensor<2x24x8x48xf32> {
    // CHECK-LABEL: func.func @repro
    // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[2, 12, 8, 48]>
    // CHECK: %[[R1:.*]] = "tfl.reshape"(%arg0, %[[CST]])
    // CHECK: %[[R2:.*]] = "tfl.reshape"(%arg1, %[[CST]])
    // CHECK: %[[CONCAT:.*]] = "tfl.concatenation"(%[[R1]], %[[R2]]) <{axis = 1 : i32
    // CHECK: return %[[CONCAT]]
    %axis_size = "tfl.pseudo_const"() <{value = dense<[2, 24, 8, 48]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %0 = "tfl.concatenation"(%arg0, %arg1) <{axis = 1 : i32, fused_activation_function = "NONE"}> : (tensor<2x1x12x8x48xf32>, tensor<2x1x12x8x48xf32>) -> tensor<2x2x12x8x48xf32>
    %1 = "tfl.reshape"(%0, %axis_size) : (tensor<2x2x12x8x48xf32>, tensor<4xi32>) -> tensor<2x24x8x48xf32>
    func.return %1 : tensor<2x24x8x48xf32>
  }

  func.func @repro_pad(%arg0: tensor<1x8x2x12x13xf32>) -> tensor<1x8x2x12x25xf32> {
    // CHECK-LABEL: func.func @repro_pad
    // CHECK: %[[R_IN:.*]] = "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x8x2x12x13xf32>, tensor<4xi32>) -> tensor<8x2x12x13xf32>
    // CHECK: %[[PAD:.*]] = "tfl.pad"(%[[R_IN]], {{.*}}) : (tensor<8x2x12x13xf32>, tensor<4x2xi32>) -> tensor<8x2x12x25xf32>
    // CHECK: %[[R_OUT:.*]] = "tfl.reshape"(%[[PAD]], {{.*}}) : (tensor<8x2x12x25xf32>, tensor<5xi32>) -> tensor<1x8x2x12x25xf32>
    %paddings = "tfl.pseudo_const"() <{value = dense<[[0, 0], [0, 0], [0, 0], [0, 0], [0, 12]]> : tensor<5x2xi32>}> : () -> tensor<5x2xi32>
    %0 = "tfl.pad"(%arg0, %paddings) : (tensor<1x8x2x12x13xf32>, tensor<5x2xi32>) -> tensor<1x8x2x12x25xf32>
    func.return %0 : tensor<1x8x2x12x25xf32>
  }

  func.func @repro_select(%arg0: tensor<1x1x2x12x24xi1>, %arg1: tensor<f32>, %arg2: tensor<1x8x2x12x24xf32>) -> tensor<1x8x2x12x24xf32> {
    // CHECK-LABEL: func.func @repro_select
    // CHECK: %[[R_COND:.*]] = "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x1x2x12x24xi1>, tensor<4xi32>) -> tensor<1x2x12x24xi1>
    // CHECK: %[[R_FALSE:.*]] = "tfl.reshape"(%arg2, {{.*}}) : (tensor<1x8x2x12x24xf32>, tensor<4xi32>) -> tensor<8x2x12x24xf32>
    // CHECK: %[[SELECT:.*]] = "tfl.select_v2"(%[[R_COND]], %arg1, %[[R_FALSE]])
    // CHECK: %[[R_OUT:.*]] = "tfl.reshape"(%[[SELECT]], {{.*}})
    // CHECK: return %[[R_OUT]]
    %0 = "tfl.select_v2"(%arg0, %arg1, %arg2) : (tensor<1x1x2x12x24xi1>, tensor<f32>, tensor<1x8x2x12x24xf32>) -> tensor<1x8x2x12x24xf32>
    func.return %0 : tensor<1x8x2x12x24xf32>
  }

  func.func @repro_softmax(%arg0: tensor<1x8x2x12x24xf32>) -> tensor<1x8x2x12x24xf32> {
    // CHECK-LABEL: func.func @repro_softmax
    // CHECK: %[[R_IN:.*]] = "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x8x2x12x24xf32>, tensor<4xi32>) -> tensor<8x2x12x24xf32>
    // CHECK: %[[SOFTMAX:.*]] = "tfl.softmax"(%[[R_IN]])
    // CHECK: %[[R_OUT:.*]] = "tfl.reshape"(%[[SOFTMAX]], {{.*}})
    // CHECK: return %[[R_OUT]]
    %0 = "tfl.softmax"(%arg0) <{beta = 1.0 : f32}> : (tensor<1x8x2x12x24xf32>) -> tensor<1x8x2x12x24xf32>
    func.return %0 : tensor<1x8x2x12x24xf32>
  }

  // CHECK-LABEL: reduce_slice_rank_generalized
  func.func @reduce_slice_rank_generalized(%arg0: tensor<4x1x12x8x48xf32>) -> tensor<2x1x12x8x48xf32> {
    %cst_39 = arith.constant dense<0> : tensor<5xi32>
    %cst_40 = arith.constant dense<[2, 1, 12, 8, 48]> : tensor<5xi32>
    // CHECK-DAG: %[[RES_SHAPE:.*]] = "tfl.pseudo_const"() <{value = dense<[2, 1, 12, 8, 48]> : tensor<5xi32>}>
    // CHECK-DAG: %[[BEGIN:.*]] = "tfl.pseudo_const"() <{value = dense<0> : tensor<4xi32>}>
    // CHECK-DAG: %[[SIZE:.*]] = "tfl.pseudo_const"() <{value = dense<[2, 12, 8, 48]> : tensor<4xi32>}>
    // CHECK-DAG: %[[NEW_IN_SHAPE:.*]] = "tfl.pseudo_const"() <{value = dense<[4, 12, 8, 48]> : tensor<4xi32>}>
    // CHECK: %[[RESHAPE_IN:.*]] = "tfl.reshape"(%arg0, %[[NEW_IN_SHAPE]]) : (tensor<4x1x12x8x48xf32>, tensor<4xi32>) -> tensor<4x12x8x48xf32>
    // CHECK: %[[SLICE:.*]] = "tfl.slice"(%[[RESHAPE_IN]], %[[BEGIN]], %[[SIZE]]) : (tensor<4x12x8x48xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<2x12x8x48xf32>
    // CHECK: %[[RESHAPE_OUT:.*]] = "tfl.reshape"(%[[SLICE]], %[[RES_SHAPE]]) : (tensor<2x12x8x48xf32>, tensor<5xi32>) -> tensor<2x1x12x8x48xf32>
    // CHECK: return %[[RESHAPE_OUT]]
    %0 = "tfl.slice"(%arg0, %cst_39, %cst_40) : (tensor<4x1x12x8x48xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<2x1x12x8x48xf32>
    func.return %0 : tensor<2x1x12x8x48xf32>
  }

  // CHECK-LABEL: dequantize_rank_reduction
  func.func @dequantize_rank_reduction(%arg0: tensor<1x8x17x128x24x!quant.uniform<i8:f32, 0.13367551565170288>>) -> tensor<1x8x17x128x24xf32> {
    // CHECK-DAG: %[[CST_OUT:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 8, 17, 128, 24]> : tensor<5xi32>}>
    // CHECK-DAG: %[[CST_IN:.*]] = "tfl.pseudo_const"() <{value = dense<[8, 17, 128, 24]> : tensor<4xi32>}>
    // CHECK: %[[RESHAPE_IN:.*]] = "tfl.reshape"(%arg0, %[[CST_IN]])
    // CHECK: %[[DEQUANT:.*]] = "tfl.dequantize"(%[[RESHAPE_IN]])
    // CHECK: %[[RESHAPE_OUT:.*]] = "tfl.reshape"(%[[DEQUANT]], %[[CST_OUT]])
    // CHECK: return %[[RESHAPE_OUT]]
    %0 = "tfl.dequantize"(%arg0) : (tensor<1x8x17x128x24x!quant.uniform<i8:f32, 0.13367551565170288>>) -> tensor<1x8x17x128x24xf32>
    func.return %0 : tensor<1x8x17x128x24xf32>
  }
}
