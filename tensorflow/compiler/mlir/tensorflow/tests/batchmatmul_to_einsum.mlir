// RUN: tf-opt %s -tf-batch-matmul-to-tf-einsum | FileCheck %s --dump-input-on-failure

func @test_batch_matmul_to_einsum(%arg0: tensor<1x2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x2x4xf32> {
  // CHECK-LABEL: test_batch_matmul_to_einsum
  // CHECK: "tf.Einsum"(%arg0, %arg1) {equation = "...mk,...kn->...mn"} : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  return %0: tensor<1x2x4xf32>
}

func @test_batch_matmulV2_to_einsum(%arg0: tensor<1x2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x2x4xf32> {
  // CHECK-LABEL: test_batch_matmulV2_to_einsum
  // CHECK: "tf.Einsum"(%arg0, %arg1) {equation = "...mk,...kn->...mn"} : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  return %0: tensor<1x2x4xf32>
}

func @test_batch_matmul_adj_to_einsum(%arg0: tensor<1x2x3xf32>, %arg1: tensor<4x3xf32>) -> tensor<1x2x4xf32> {
  // CHECK-LABEL: test_batch_matmul_adj_to_einsum
  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[RES_TRANS:[0-9]*]] = "tf.Transpose"(%arg1, %[[RES_PERM]]) : (tensor<4x3xf32>, tensor<2xi32>) -> tensor<3x4xf32>
  // CHECK: %[[RES_EINSUM:[0-9]*]] = "tf.Einsum"(%arg0, %[[RES_TRANS]]) {equation = "...mk,...kn->...mn"} : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  // CHECK: return %[[RES_EINSUM]] : tensor<1x2x4xf32>
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = true} : (tensor<1x2x3xf32>, tensor<4x3xf32>) -> tensor<1x2x4xf32>
  return %0: tensor<1x2x4xf32>
}

func @test_batch_matmulV2_adj_to_einsum(%arg0: tensor<1x3x2xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x2x4xf32> {
  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK: %[[RES_TRANS:[0-9]*]] = "tf.Transpose"(%arg0, %[[RES_PERM]]) : (tensor<1x3x2xf32>, tensor<3xi32>) -> tensor<1x2x3xf32>
  // CHECK: %[[RES_EINSUM:[0-9]*]] = "tf.Einsum"(%[[RES_TRANS]], %arg1) {equation = "...mk,...kn->...mn"} : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  // CHECK: return %[[RES_EINSUM]] : tensor<1x2x4xf32>
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = false} : (tensor<1x3x2xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  return %0: tensor<1x2x4xf32>
}
