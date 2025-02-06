// Run optimize-batch-matmul pass only and check the results.
// RUN: tf-opt %s -tfl-optimize-batch-matmul | FileCheck %s

// CHECK-LABEL: FuseTransposeFCRhsToBatchMatmul
func.func @FuseTransposeFCRhsToBatchMatmul(%arg0: tensor<16x1024xf32>, %arg1: tensor<1024x128xf32>, %arg2: none) -> tensor<16x128xf32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tfl.transpose"(%arg1, %cst) : (tensor<1024x128xf32>, tensor<2xi32>) -> tensor<128x1024xf32>
  // CHECK: "tfl.batch_matmul"(%arg0, %arg1)
  %1 = "tfl.fully_connected"(%arg0, %0, %arg2) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<16x1024xf32>, tensor<128x1024xf32>, none) -> tensor<16x128xf32>
  func.return %1 : tensor<16x128xf32>
  // CHECK: return %0 : tensor<16x128xf32>
}

// CHECK-LABEL: FuseTransposeFCLhsToBatchMatmul
func.func @FuseTransposeFCLhsToBatchMatmul(%arg0: tensor<1024x4xf32>, %arg1: tensor<8x1024xf32>, %arg2: tensor<4x256xf32>) -> tensor<8x256xf32> {
  %cst_0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst_1 = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.transpose"(%arg0, %cst_0) : (tensor<1024x4xf32>, tensor<2xi32>) -> tensor<4x1024xf32>
  // CHECK: %[[RES0:.*]] = "tfl.batch_matmul"(%arg1, %arg0) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<8x1024xf32>, tensor<1024x4xf32>) -> tensor<8x4xf32>
  %1 = "tfl.fully_connected"(%0, %arg1, %cst_1) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x1024xf32>, tensor<8x1024xf32>, none) -> tensor<4x8xf32>
  // CHECK: %[[RES1:.*]] = "tfl.batch_matmul"(%[[RES0]], %arg2) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<8x4xf32>, tensor<4x256xf32>) -> tensor<8x256xf32>
  %2 = "tfl.batch_matmul"(%1, %arg2) {adj_x = true, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x8xf32>, tensor<4x256xf32>) -> tensor<8x256xf32>
  func.return %2 : tensor<8x256xf32>
  // CHECK: return %[[RES1]] : tensor<8x256xf32>
}

// CHECK-LABEL: Batchmatmul2Fullyconnected
// CHECK-NOT: "tfl.batch_matmul"
func.func @Batchmatmul2Fullyconnected(%arg0: tensor<4x128x2xf32>) -> (tensor<4x128x1xf32>) {
  %0 = arith.constant dense<[[1.0], [2.0]]> : tensor<2x1xf32>
  %1 = "tfl.batch_matmul"(%arg0, %0) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x128x2xf32>, tensor<2x1xf32>) -> tensor<4x128x1xf32>
  func.return %1 : tensor<4x128x1xf32>
  // CHECK-NEXT: %[[CONST_WEIGHT:.*]] = arith.constant
  // CHECK-SAME: [1.000000e+00, 2.000000e+00]
  // CHECK-SAME: tensor<1x2xf32>
  // CHECK: %[[FC_RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONST_WEIGHT]]
  // CHECK-SAME: <{fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<4x128x2xf32>, tensor<1x2xf32>, none) -> tensor<4x128x1xf32>
  // CHECK-NEXT: return %[[FC_RES]]
}

// CHECK-LABEL: Batchmatmul2FullyconnectedAdjy
// CHECK-NOT: "tfl.batch_matmul"
func.func @Batchmatmul2FullyconnectedAdjy(%arg0: tensor<4x128x2xf32>) -> (tensor<4x128x1xf32>) {
  %0 = arith.constant dense<[[1.0, 2.0]]> : tensor<1x2xf32>
  %1 = "tfl.batch_matmul"(%arg0, %0) {adj_x = false, adj_y = true, asymmetric_quantize_inputs = false} : (tensor<4x128x2xf32>, tensor<1x2xf32>) -> tensor<4x128x1xf32>
  func.return %1 : tensor<4x128x1xf32>
  // CHECK: %[[CONST_WEIGHT:.*]] = arith.constant
  // CHECK-SAME: [1.000000e+00, 2.000000e+00]
  // CHECK-SAME: tensor<1x2xf32>
  // CHECK: %[[FC_RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONST_WEIGHT]]
  // CHECK-SAME: <{fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<4x128x2xf32>, tensor<1x2xf32>, none) -> tensor<4x128x1xf32>
  // CHECK-NEXT: return %[[FC_RES]]
}

// CHECK-LABEL: Batchmatmul2FullyconnectedAdjx
// CHECK-NOT: "tfl.batch_matmul"
func.func @Batchmatmul2FullyconnectedAdjx(%arg0: tensor<4x2x128xf32>) -> (tensor<4x128x1xf32>) {
  %0 = arith.constant dense<[[1.0], [2.0]]> : tensor<2x1xf32>
  %1 = "tfl.batch_matmul"(%arg0, %0) {adj_x = true, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x2x128xf32>, tensor<2x1xf32>) -> tensor<4x128x1xf32>
  func.return %1 : tensor<4x128x1xf32>

  // CHECK: %[[TRANSPOSED_X:.*]] = "tfl.transpose"
  // CHECK-SAME: (tensor<4x2x128xf32>, tensor<3xi32>) -> tensor<4x128x2xf32>
  // CHECK-NEXT: %[[FC_RES:.*]] = "tfl.fully_connected"(%[[TRANSPOSED_X]]
  // CHECK-SAME: <{fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<4x128x2xf32>, tensor<1x2xf32>, none) -> tensor<4x128x1xf32>
  // CHECK-NEXT: return %[[FC_RES]]
}

// CHECK-LABEL: Batchmatmul2FullyconnectedBatchedY
// BMM can be converted to FC only if we have constant weight with rank 2.
// CHECK-NOT: "tfl.fully_connected"
func.func @Batchmatmul2FullyconnectedBatchedY(%arg0: tensor<4x128x2xf32>) -> (tensor<4x128x1xf32>) {
  %0 = arith.constant dense<42.> : tensor<4x2x1xf32>
  %1 = "tfl.batch_matmul"(%arg0, %0) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x128x2xf32>, tensor<4x2x1xf32>) -> tensor<4x128x1xf32>
  func.return %1 : tensor<4x128x1xf32>

  // CHECK: %[[BMM_RES:.*]] = "tfl.batch_matmul"
  // CHECK-NEXT: return %[[BMM_RES]]
}

// CHECK-LABEL: Batchmatmul2FullyconnectedTransposedY
func.func @Batchmatmul2FullyconnectedTransposedY(%arg0: tensor<4x128x2xf32>) -> (tensor<4x128x1xf32>) {
  %0 = arith.constant dense<[[1.0], [2.0]]> : tensor<2x1xf32>
  %1 = "tfl.batch_matmul"(%arg0, %0) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x128x2xf32>, tensor<2x1xf32>) -> tensor<4x128x1xf32>
  func.return %1 : tensor<4x128x1xf32>
  // CHECK: %[[CONST_WEIGHT:.*]] = arith.constant
  // CHECK-SAME: [1.000000e+00, 2.000000e+00]
  // CHECK-SAME: tensor<1x2xf32>
  // CHECK: %[[FC_RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONST_WEIGHT]]
  // CHECK-SAME: <{fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<4x128x2xf32>, tensor<1x2xf32>, none) -> tensor<4x128x1xf32>
  // CHECK-NEXT: return %[[FC_RES]]
}

// CHECK-LABEL: Batchmatmul2FullyconnectedNonConstY
// BMM can be converted to FC only if we have constant weight with rank 2.
// CHECK-NOT: "tfl.fully_connected"
func.func @Batchmatmul2FullyconnectedNonConstY(%arg0: tensor<4x128x2xf32>, %arg1: tensor<2x1xf32>) -> (tensor<4x128x1xf32>) {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x128x2xf32>, tensor<2x1xf32>) -> tensor<4x128x1xf32>
  func.return %0 : tensor<4x128x1xf32>

  // CHECK: %[[BMM_RES:.*]] = "tfl.batch_matmul"
  // CHECK-NEXT: return %[[BMM_RES]]
}

// CHECK-LABEL: Batchmatmul2FullyconnectedQDQ
// CHECK-NOT: "tfl.batch_matmul"
func.func @Batchmatmul2FullyconnectedQDQ(%arg0: tensor<4x128x2xf32>, %arg1: tensor<2x1xf32>) -> (tensor<4x128x1xf32>) {
  %0 = arith.constant dense<[[1.0], [2.0]]> : tensor<2x1xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x1x!quant.uniform<i8:f32, 0.024986599940879671:92>>} : (tensor<2x1xf32>) -> tensor<2x1x!quant.uniform<i8:f32, 0.024986599940879671:92>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x1x!quant.uniform<i8:f32, 0.024986599940879671:92>>) -> tensor<2x1xf32>
  %3 = "tfl.batch_matmul"(%arg0, %2) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x128x2xf32>, tensor<2x1xf32>) -> tensor<4x128x1xf32>
  func.return %3 : tensor<4x128x1xf32>
  // CHECK: %[[TRANSPOSED_X:.*]] = "tfl.transpose"
  // CHECK-SAME: (tensor<2x1xf32>, tensor<2xi32>) -> tensor<1x2xf32>
  // CHECK: %[[FC_RES:.*]] = "tfl.fully_connected"(%arg0, %[[TRANSPOSED_X]]
  // CHECK-SAME: <{fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<4x128x2xf32>, tensor<1x2xf32>, none) -> tensor<4x128x1xf32>
  // CHECK-NEXT: return %[[FC_RES]]
}

// CHECK-LABEL: BatchmatmulToReduceSumI32
// CHECK-NOT: "tfl.batch_matmul"
func.func @BatchmatmulToReduceSumI32(%arg0: tensor<1x16384x257xi32>) -> (tensor<1x1x257xi32>) {
  %0 = arith.constant dense<1> : tensor<1x1x16384xi32>
  %1 = "tfl.batch_matmul"(%0, %arg0) {adj_x = false, adj_y = false} : (tensor<1x1x16384xi32>, tensor<1x16384x257xi32>) -> tensor<1x1x257xi32>
  func.return %1 : tensor<1x1x257xi32>
  // CHECK: %[[CONST_DIM:.*]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[RED:.*]] = "tfl.sum"(%arg0, %[[CONST_DIM]]) <{keep_dims = true}> : (tensor<1x16384x257xi32>, tensor<1xi32>) -> tensor<1x1x257xi32>
}

// CHECK-LABEL: BatchmatmulToReduceSumF32
// CHECK-NOT: "tfl.batch_matmul"
func.func @BatchmatmulToReduceSumF32(%arg0: tensor<1x16384x257xf32>) -> (tensor<1x1x257xf32>) {
  %0 = arith.constant dense<1.0> : tensor<1x1x16384xf32>
  %1 = "tfl.batch_matmul"(%0, %arg0) {adj_x = false, adj_y = false} : (tensor<1x1x16384xf32>, tensor<1x16384x257xf32>) -> tensor<1x1x257xf32>
  func.return %1 : tensor<1x1x257xf32>
  // CHECK: %[[CONST_DIM:.*]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK: %[[RED:.*]] = "tfl.sum"(%arg0, %[[CONST_DIM]]) <{keep_dims = true}> : (tensor<1x16384x257xf32>, tensor<1xi32>) -> tensor<1x1x257xf32>
}
