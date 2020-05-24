// RUN: tf-opt %s -tf-batch-matmul-to-tf-einsum | FileCheck %s --dump-input-on-failure

func @test_batch_matmul_to_einsum(%arg0: tensor<1x2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x2x4xf32> {
  // CHECK-LABEL: test_batch_matmul_to_einsum
  // CHECK: "tf.Einsum"(%arg0, %arg1) {equation = "...mk,...kn->...mn"} : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x2x3xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  return %0: tensor<1x2x4xf32>
}

func @test_batch_matmul_broadcast_to_einsum(%arg0: tensor<2x2x4xf32>, %arg1: tensor<2x4x2xf32>) -> tensor<2x2x2xf32> {
  // CHECK-LABEL: test_batch_matmul_broadcast_to_einsum
  // CHECK: "tf.Einsum"(%arg0, %arg1) {equation = "...mk,...kn->...mn"} : (tensor<2x2x4xf32>, tensor<2x4x2xf32>) -> tensor<2x2x2xf32>
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<2x2x4xf32>, tensor<2x4x2xf32>) -> tensor<2x2x2xf32>
  return %0: tensor<2x2x2xf32>
}

func @test_batch_matmul_dynamic_shape_both_arg_to_einsum(%arg0: tensor<1x2x?xf32>, %arg1: tensor<?x4xf32>) -> tensor<1x2x4xf32> {
  // CHECK-LABEL: test_batch_matmul_dynamic_shape_both_arg_to_einsum
  // CHECK: "tf.Einsum"(%arg0, %arg1) {equation = "...mk,...kn->...mn"} : (tensor<1x2x?xf32>, tensor<?x4xf32>) -> tensor<1x2x4xf32>
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x2x?xf32>, tensor<?x4xf32>) -> tensor<1x2x4xf32>
  return %0: tensor<1x2x4xf32>
}

func @test_batch_matmul_dynamic_shape_one_arg_to_einsum(%arg0: tensor<1x2x?xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x2x4xf32> {
  // CHECK-LABEL: test_batch_matmul_dynamic_shape_one_arg_to_einsum
  // CHECK: "tf.Einsum"(%arg0, %arg1) {equation = "...mk,...kn->...mn"} : (tensor<1x2x?xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x2x?xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  return %0: tensor<1x2x4xf32>
}

func @test_batch_matmul_adj_to_einsum(%arg0: tensor<1x2x3xf32>, %arg1: tensor<4x3xf32>) -> tensor<1x2x4xf32> {
  // CHECK-LABEL: test_batch_matmul_adj_to_einsum
  // CHECK: %[[RES_EINSUM:[0-9]*]] = "tf.Einsum"(%arg0, %arg1) {equation = "...mk,...nk->...mn"} : (tensor<1x2x3xf32>, tensor<4x3xf32>) -> tensor<1x2x4xf32>
  // CHECK: return %[[RES_EINSUM]] : tensor<1x2x4xf32>
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = true} : (tensor<1x2x3xf32>, tensor<4x3xf32>) -> tensor<1x2x4xf32>
  return %0: tensor<1x2x4xf32>
}

func @test_batch_matmulV2_adj_to_einsum(%arg0: tensor<1x3x2xf32>, %arg1: tensor<3x4xf32>) -> tensor<1x2x4xf32> {
  // CHECK: %[[RES_EINSUM:[0-9]*]] = "tf.Einsum"(%arg0, %arg1) {equation = "...km,...kn->...mn"} : (tensor<1x3x2xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  // CHECK: return %[[RES_EINSUM]] : tensor<1x2x4xf32>
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = false} : (tensor<1x3x2xf32>, tensor<3x4xf32>) -> tensor<1x2x4xf32>
  return %0: tensor<1x2x4xf32>
}
