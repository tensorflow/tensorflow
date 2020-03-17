// RUN: tf-opt -split-input-file -verify-diagnostics -tf-einsum %s | FileCheck %s

func @einsum_basic(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ijk,ikm->ijm"}: (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  return %0 : tensor<3x4x6xf32>
  // CHECK-LABEL: einsum_basic
  // CHECK: "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
}

func @einsum_4D(%arg0: tensor<2x5x7x3xf32>, %arg1: tensor<2x4x7x3xf32>) -> tensor<2x7x5x4xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "bfnh,btnh->bnft"}: (tensor<2x5x7x3xf32>, tensor<2x4x7x3xf32>) -> tensor<2x7x5x4xf32>
  return %0 : tensor<2x7x5x4xf32>
  // CHECK-LABEL: einsum_4D
  // CHECK: %[[cst:.*]] = constant dense<[0, 2, 1, 3]> : tensor<4xi32>
  // CHECK: %[[cst_1:.*]] = constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg0, %[[cst]]) : (tensor<2x5x7x3xf32>, tensor<4xi32>) -> tensor<2x7x5x3xf32>
  // CHECK: %[[v1:.*]] = "tf.Transpose"(%arg1, %[[cst_1]]) : (tensor<2x4x7x3xf32>, tensor<4xi32>) -> tensor<2x7x3x4xf32>
  // CHECK: "tf.BatchMatMulV2"(%[[v0]], %[[v1]]) {adj_x = false, adj_y = false} : (tensor<2x7x5x3xf32>, tensor<2x7x3x4xf32>) -> tensor<2x7x5x4xf32>
}

func @einsum_matrixdotprod(%arg0: tensor<2x5x7x3xf32>, %arg1: tensor<7x3x4xf32>) -> tensor<2x5x4xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "bfnd,ndh->bfh"}: (tensor<2x5x7x3xf32>, tensor<7x3x4xf32>) -> tensor<2x5x4xf32>
  return %0 : tensor<2x5x4xf32>
  // CHECK-LABEL: einsum_matrixdotprod
  // CHECK: %[[cst:.*]] = constant dense<[2, 5, 21]> : tensor<3xi64>
  // CHECK: %[[cst_1:.*]] = constant dense<[21, 4]> : tensor<2xi64>
  // CHECK: %[[v0:.*]] = "tf.Reshape"(%arg0, %[[cst]]) : (tensor<2x5x7x3xf32>, tensor<3xi64>) -> tensor<2x5x21xf32>
  // CHECK: %[[v1:.*]] = "tf.Reshape"(%arg1, %[[cst_1]]) : (tensor<7x3x4xf32>, tensor<2xi64>) -> tensor<21x4xf32>
  // CHECK: "tf.BatchMatMulV2"(%[[v0]], %[[v1]]) {adj_x = false, adj_y = false} : (tensor<2x5x21xf32>, tensor<21x4xf32>) -> tensor<2x5x4xf32>
}

func @einsum_reshapetail(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5x6x2xf32>) -> tensor<3x4x6x2xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "bfd,dnh->bfnh"}: (tensor<3x4x5xf32>, tensor<5x6x2xf32>) -> tensor<3x4x6x2xf32>
  return %0 : tensor<3x4x6x2xf32>
  // CHECK-LABEL: einsum_reshapetail
  // CHECK: %[[cst:.*]] = constant dense<[5, 12]> : tensor<2xi64>
  // CHECK: %[[cst_1:.*]] = constant dense<[3, 4, 6, 2]> : tensor<4xi64>
  // CHECK: %[[v0:.*]] = "tf.Reshape"(%arg1, %[[cst]]) : (tensor<5x6x2xf32>, tensor<2xi64>) -> tensor<5x12xf32>
  // CHECK: %[[v1:.*]] = "tf.BatchMatMulV2"(%arg0, %[[v0]]) {adj_x = false, adj_y = false} : (tensor<3x4x5xf32>, tensor<5x12xf32>) -> tensor<3x4x12xf32>
  // CHECK: %[[v2:.*]] = "tf.Reshape"(%[[v1]], %[[cst_1]]) : (tensor<3x4x12xf32>, tensor<4xi64>) -> tensor<3x4x6x2xf32>
  // CHECK: return %[[v2]] : tensor<3x4x6x2xf32>
}

func @einsum_no_match(%arg0: tensor<4x5xf32>, %arg1: tensor<5xf32>) -> tensor<4xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ij,j->i"}: (tensor<4x5xf32>, tensor<5xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
// CHECK-LABEL: einsum_no_match
// CHECK: %[[v0:.*]] = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ij,j->i"} : (tensor<4x5xf32>, tensor<5xf32>) -> tensor<4xf32>
// CHECK: return %[[v0]]
}
func @einsum_illegal_no_match(%arg0: tensor<4x5xf32>, %arg1: tensor<5xf32>) -> tensor<4xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ij,?zw->kq->i"}: (tensor<4x5xf32>, tensor<5xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
// CHECK-LABEL: einsum_illegal_no_match
// CHECK: %[[v0:.*]] = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ij,?zw->kq->i"} : (tensor<4x5xf32>, tensor<5xf32>) -> tensor<4xf32>
// CHECK: return %[[v0]]
}
