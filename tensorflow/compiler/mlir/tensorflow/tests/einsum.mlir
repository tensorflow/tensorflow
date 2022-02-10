// RUN: tf-opt -split-input-file -verify-diagnostics -tf-einsum %s | FileCheck %s

func @einsum_basic(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ijk,ikm->ijm"}: (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  return %0 : tensor<3x4x6xf32>
  // CHECK-LABEL: einsum_basic
  // CHECK: "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
}

func @einsum_matmul(%arg0: tensor<7x9xf32>, %arg1: tensor<9x5xf32>) -> tensor<7x5xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ae,ed->ad"}: (tensor<7x9xf32>, tensor<9x5xf32>) -> tensor<7x5xf32>
  return %0 : tensor<7x5xf32>
  // CHECK-LABEL: einsum_matmul
  // CHECK: %[[v0:.*]] = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<7x9xf32>, tensor<9x5xf32>) -> tensor<7x5xf32>
  // CHECK: return %[[v0]] : tensor<7x5xf32>
}

func @einsum_broadcast(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ijk,km->ijm"}: (tensor<3x4x5xf32>, tensor<5x6xf32>) -> tensor<3x4x6xf32>
  return %0 : tensor<3x4x6xf32>
  // CHECK-LABEL: einsum_broadcast
  // CHECK: "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<3x4x5xf32>, tensor<5x6xf32>) -> tensor<3x4x6xf32>
}

func @einsum_broadcast4(%arg0: tensor<3x4x5x6x7xf32>, %arg1: tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "abcdh,hg->abcdg"}: (tensor<3x4x5x6x7xf32>, tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32>
  return %0 : tensor<3x4x5x6x8xf32>
  // CHECK-LABEL: einsum_broadcast4
  // CHECK: "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<3x4x5x6x7xf32>, tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32>
}

func @einsum_reducesum(%arg0: tensor<2x5x7xf32>, %arg1: tensor<5x2xf32>) -> tensor<5x7xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "lbh,bl->bh"}: (tensor<2x5x7xf32>, tensor<5x2xf32>) -> tensor<5x7xf32>
  return %0 : tensor<5x7xf32>
  // CHECK-LABEL: einsum_reducesum
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[5, 2, 1]> : tensor<3xi64>
  // CHECK-DAG: %[[cst_2:.*]] = arith.constant dense<[5, 7]> : tensor<2xi64>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg0, %[[cst]]) : (tensor<2x5x7xf32>, tensor<3xi32>) -> tensor<5x7x2xf32>
  // CHECK: %[[v1:.*]] = "tf.Reshape"(%arg1, %[[cst_1]]) : (tensor<5x2xf32>, tensor<3xi64>) -> tensor<5x2x1xf32>
  // CHECK: %[[v2:.*]] = "tf.BatchMatMulV2"(%[[v0]], %[[v1]]) {adj_x = false, adj_y = false} : (tensor<5x7x2xf32>, tensor<5x2x1xf32>) -> tensor<5x7x1xf32>
  // CHECK: %[[v3:.*]] = "tf.Reshape"(%[[v2]], %[[cst_2]]) : (tensor<5x7x1xf32>, tensor<2xi64>) -> tensor<5x7xf32>
  // CHECK: return %[[v3:.*]] : tensor<5x7xf32>
}

func @einsum_transpose_matmul(%arg0: tensor<2x5x7xf32>, %arg1: tensor<5x3x2xf32>) -> tensor<5x3x7xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "lbh,bkl->bkh"}: (tensor<2x5x7xf32>, tensor<5x3x2xf32>) -> tensor<5x3x7xf32>
  return %0 : tensor<5x3x7xf32>
  // CHECK-LABEL: einsum_transpose_matmul
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
  // CHECK-DAG: %[[cst_0:.*]] = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg0, %[[cst]]) : (tensor<2x5x7xf32>, tensor<3xi32>) -> tensor<5x7x2xf32>
  // CHECK: %[[v1:.*]] = "tf.Transpose"(%arg1, %[[cst_0]]) : (tensor<5x3x2xf32>, tensor<3xi32>) -> tensor<5x2x3xf32>
  // CHECK: %[[v2:.*]] = "tf.BatchMatMulV2"(%[[v0]], %[[v1]]) {adj_x = false, adj_y = false} : (tensor<5x7x2xf32>, tensor<5x2x3xf32>) -> tensor<5x7x3xf32>
  // CHECK: %[[v3:.*]] = "tf.Transpose"(%[[v2]], %[[cst_0]]) : (tensor<5x7x3xf32>, tensor<3xi32>) -> tensor<5x3x7xf32>
}

func @einsum_4D(%arg0: tensor<2x5x7x3xf32>, %arg1: tensor<2x4x7x3xf32>) -> tensor<2x7x5x4xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "bfnh,btnh->bnft"}: (tensor<2x5x7x3xf32>, tensor<2x4x7x3xf32>) -> tensor<2x7x5x4xf32>
  return %0 : tensor<2x7x5x4xf32>
  // CHECK-LABEL: einsum_4D
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg0, %[[cst]]) : (tensor<2x5x7x3xf32>, tensor<4xi32>) -> tensor<2x7x5x3xf32>
  // CHECK: %[[v1:.*]] = "tf.Transpose"(%arg1, %[[cst_1]]) : (tensor<2x4x7x3xf32>, tensor<4xi32>) -> tensor<2x7x3x4xf32>
  // CHECK: "tf.BatchMatMulV2"(%[[v0]], %[[v1]]) {adj_x = false, adj_y = false} : (tensor<2x7x5x3xf32>, tensor<2x7x3x4xf32>) -> tensor<2x7x5x4xf32>
}

func @einsum_matrixdotprod(%arg0: tensor<2x5x7x3xf32>, %arg1: tensor<7x3x4xf32>) -> tensor<2x5x4xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "bfnd,ndh->bfh"}: (tensor<2x5x7x3xf32>, tensor<7x3x4xf32>) -> tensor<2x5x4xf32>
  return %0 : tensor<2x5x4xf32>
  // CHECK-LABEL: einsum_matrixdotprod
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[2, 5, 21]> : tensor<3xi64>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[21, 4]> : tensor<2xi64>
  // CHECK: %[[v0:.*]] = "tf.Reshape"(%arg0, %[[cst]]) : (tensor<2x5x7x3xf32>, tensor<3xi64>) -> tensor<2x5x21xf32>
  // CHECK: %[[v1:.*]] = "tf.Reshape"(%arg1, %[[cst_1]]) : (tensor<7x3x4xf32>, tensor<2xi64>) -> tensor<21x4xf32>
  // CHECK: "tf.BatchMatMulV2"(%[[v0]], %[[v1]]) {adj_x = false, adj_y = false} : (tensor<2x5x21xf32>, tensor<21x4xf32>) -> tensor<2x5x4xf32>
}

func @einsum_reshapetail(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5x6x2xf32>) -> tensor<3x4x6x2xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "bfd,dnh->bfnh"}: (tensor<3x4x5xf32>, tensor<5x6x2xf32>) -> tensor<3x4x6x2xf32>
  return %0 : tensor<3x4x6x2xf32>
  // CHECK-LABEL: einsum_reshapetail
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[5, 12]> : tensor<2xi64>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[3, 4, 6, 2]> : tensor<4xi64>
  // CHECK: %[[v0:.*]] = "tf.Reshape"(%arg1, %[[cst]]) : (tensor<5x6x2xf32>, tensor<2xi64>) -> tensor<5x12xf32>
  // CHECK: %[[v1:.*]] = "tf.BatchMatMulV2"(%arg0, %[[v0]]) {adj_x = false, adj_y = false} : (tensor<3x4x5xf32>, tensor<5x12xf32>) -> tensor<3x4x12xf32>
  // CHECK: %[[v2:.*]] = "tf.Reshape"(%[[v1]], %[[cst_1]]) : (tensor<3x4x12xf32>, tensor<4xi64>) -> tensor<3x4x6x2xf32>
  // CHECK: return %[[v2]] : tensor<3x4x6x2xf32>
}

func @einsum_reduceddim(%arg0: tensor<2x5x7xf32>, %arg1: tensor<2x5x7x3xf32>) -> tensor<2x5x3xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "bin,binj->bij"}: (tensor<2x5x7xf32>, tensor<2x5x7x3xf32>) -> tensor<2x5x3xf32>
  return %0 : tensor<2x5x3xf32>
  // CHECK-LABEL: einsum_reduceddim
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[2, 5, 1, 7]> : tensor<4xi64>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[2, 5, 3]> : tensor<3xi64>
  // CHECK: %[[v0:.*]] = "tf.Reshape"(%arg0, %[[cst]]) : (tensor<2x5x7xf32>, tensor<4xi64>) -> tensor<2x5x1x7xf32>
  // CHECK: %[[v1:.*]] = "tf.BatchMatMulV2"(%[[v0]], %arg1) {adj_x = false, adj_y = false} : (tensor<2x5x1x7xf32>, tensor<2x5x7x3xf32>) -> tensor<2x5x1x3xf32>
  // CHECK: %[[v2:.*]] = "tf.Reshape"(%[[v1]], %[[cst_1]]) : (tensor<2x5x1x3xf32>, tensor<3xi64>) -> tensor<2x5x3xf32>
  // CHECK: return %[[v2]] : tensor<2x5x3xf32>
}

func @einsum_transposereduceddim(%arg0: tensor<2x5x7xf32>, %arg1: tensor<2x5x3x7xf32>) -> tensor<2x5x3xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "bij,binj->bin"}: (tensor<2x5x7xf32>, tensor<2x5x3x7xf32>) -> tensor<2x5x3xf32>
  return %0 : tensor<2x5x3xf32>
  // CHECK-LABEL: einsum_transposereduceddim
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[2, 5, 1, 7]> : tensor<4xi64>
  // CHECK-DAG: %[[cst_2:.*]] = arith.constant dense<[2, 5, 3]> : tensor<3xi64>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg1, %[[cst]]) : (tensor<2x5x3x7xf32>, tensor<4xi32>) -> tensor<2x5x7x3xf32>
  // CHECK: %[[v1:.*]] = "tf.Reshape"(%arg0, %[[cst_1]]) : (tensor<2x5x7xf32>, tensor<4xi64>) -> tensor<2x5x1x7xf32>
  // CHECK: %[[v2:.*]] = "tf.BatchMatMulV2"(%[[v1]], %[[v0]]) {adj_x = false, adj_y = false} : (tensor<2x5x1x7xf32>, tensor<2x5x7x3xf32>) -> tensor<2x5x1x3xf32>
  // CHECK: %[[v3:.*]] = "tf.Reshape"(%[[v2]], %[[cst_2]]) : (tensor<2x5x1x3xf32>, tensor<3xi64>) -> tensor<2x5x3xf32>
  // CHECK: return %[[v3]] : tensor<2x5x3xf32>
}

func @einsum_fourdreducelast(%arg0: tensor<2x5x7x3xf32>, %arg1: tensor<2x3x5x13xf32>) -> tensor<2x7x5x13xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "acbe,aecd->abcd"}: (tensor<2x5x7x3xf32>, tensor<2x3x5x13xf32>) -> tensor<2x7x5x13xf32>
  return %0 : tensor<2x7x5x13xf32>
  // CHECK-LABEL: einsum_fourdreducelast
  // CHECK: %[[cst:.*]] = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg1, %[[cst]]) : (tensor<2x3x5x13xf32>, tensor<4xi32>) -> tensor<2x5x3x13xf32>
  // CHECK: %[[v1:.*]] = "tf.BatchMatMulV2"(%arg0, %[[v0]]) {adj_x = false, adj_y = false} : (tensor<2x5x7x3xf32>, tensor<2x5x3x13xf32>) -> tensor<2x5x7x13xf32>
  // CHECK: %[[v2:.*]] = "tf.Transpose"(%[[v1]], %[[cst]]) : (tensor<2x5x7x13xf32>, tensor<4xi32>) -> tensor<2x7x5x13xf32>
  // CHECK: return %[[v2]] : tensor<2x7x5x13xf32>
}

func @einsum_fourdtransposeall(%arg0: tensor<2x5x7x3xf32>, %arg1: tensor<2x11x7x3xf32>) -> tensor<2x7x11x5xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "aecd,abcd->acbe"}: (tensor<2x5x7x3xf32>, tensor<2x11x7x3xf32>) -> tensor<2x7x11x5xf32>
  return %0 : tensor<2x7x11x5xf32>
  // CHECK-LABEL: einsum_fourdtransposeall
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  // CHECK-DAG: %[[cst_2:.*]] = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg0, %[[cst]]) : (tensor<2x5x7x3xf32>, tensor<4xi32>) -> tensor<2x7x5x3xf32>
  // CHECK: %[[v1:.*]] = "tf.Transpose"(%arg1, %[[cst_1]]) : (tensor<2x11x7x3xf32>, tensor<4xi32>) -> tensor<2x7x3x11xf32>
  // CHECK: %[[v2:.*]] = "tf.BatchMatMulV2"(%[[v0]], %[[v1]]) {adj_x = false, adj_y = false} : (tensor<2x7x5x3xf32>, tensor<2x7x3x11xf32>) -> tensor<2x7x5x11xf32>
  // CHECK: %[[v3:.*]] = "tf.Transpose"(%[[v2]], %[[cst_2]]) : (tensor<2x7x5x11xf32>, tensor<4xi32>) -> tensor<2x7x11x5xf32>
  // CHECK: return %[[v3]] : tensor<2x7x11x5xf32>
}

func @einsum_4d_1(%arg0: tensor<3x4x5x6xf32>, %arg1: tensor<3x7x5x6xf32>) -> tensor<3x5x4x7xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "jbki,jfki->jkbf"}: (tensor<3x4x5x6xf32>, tensor<3x7x5x6xf32>) -> tensor<3x5x4x7xf32>
  return %0 : tensor<3x5x4x7xf32>
  // CHECK-LABEL: einsum_4d_1
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg0, %[[cst:.*]]) : (tensor<3x4x5x6xf32>, tensor<4xi32>) -> tensor<3x5x4x6xf32>
  // CHECK: %[[v1:.*]] = "tf.Transpose"(%arg1, %[[cst_1]]) : (tensor<3x7x5x6xf32>, tensor<4xi32>) -> tensor<3x5x6x7xf32>
  // CHECK: %[[v2:.*]] = "tf.BatchMatMulV2"(%[[v0]], %[[v1]]) {adj_x = false, adj_y = false} : (tensor<3x5x4x6xf32>, tensor<3x5x6x7xf32>) -> tensor<3x5x4x7xf32>
  // CHECK: return %[[v2]] : tensor<3x5x4x7xf32>
}

func @einsum_no_match(%arg0: tensor<4x5x6xf32>, %arg1: tensor<5xf32>) -> tensor<4xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ijk,j->i"}: (tensor<4x5x6xf32>, tensor<5xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
// CHECK-LABEL: einsum_no_match
// CHECK: %[[v0:.*]] = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ijk,j->i"} : (tensor<4x5x6xf32>, tensor<5xf32>) -> tensor<4xf32>
// CHECK: return %[[v0]]
}

func @einsum_illegal_no_match(%arg0: tensor<4x5xf32>, %arg1: tensor<5xf32>) -> tensor<4xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ij,?zw->kq->i"}: (tensor<4x5xf32>, tensor<5xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
// CHECK-LABEL: einsum_illegal_no_match
// CHECK: %[[v0:.*]] = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ij,?zw->kq->i"} : (tensor<4x5xf32>, tensor<5xf32>) -> tensor<4xf32>
// CHECK: return %[[v0]]
}

func @batch_multilhs_einsum(%arg0: tensor<2x1x1x11xf32>, %arg1: tensor<2x11x2xf32>) -> tensor<2x1x1x2xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "BiNj,BjS->BiNS"} : (tensor<2x1x1x11xf32>, tensor<2x11x2xf32>) -> tensor<2x1x1x2xf32>
  return %0 : tensor<2x1x1x2xf32>
// CHECK-LABEL: batch_multilhs_einsum
// CHECK-DAG: %[[cst:.*]] = arith.constant dense<[2, 1, 11]> : tensor<3xi64>
// CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[2, 1, 1, 2]> : tensor<4xi64>
// CHECK: %[[v0:.*]] = "tf.Reshape"(%arg0, %[[cst]]) : (tensor<2x1x1x11xf32>, tensor<3xi64>) -> tensor<2x1x11xf32>
// CHECK: %[[v1:.*]] = "tf.BatchMatMulV2"(%[[v0]], %arg1) {adj_x = false, adj_y = false} : (tensor<2x1x11xf32>, tensor<2x11x2xf32>) -> tensor<2x1x2xf32>
// CHECK: %[[v2:.*]] = "tf.Reshape"(%[[v1]], %[[cst_1]]) : (tensor<2x1x2xf32>, tensor<4xi64>) -> tensor<2x1x1x2xf32>
// CHECK: return %[[v2]] : tensor<2x1x1x2xf32>
}

func @einsum_no_match_on_invalid_reshape_op_1(%arg0 : tensor<?x36x32xf32>, %arg1 : tensor<?x36x?x32xf32>) -> tensor<?x36x?xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {device = "", equation = "bij,binj->bin"} : (tensor<?x36x32xf32>, tensor<?x36x?x32xf32>) -> tensor<?x36x?xf32>
  return %0 : tensor<?x36x?xf32>
// CHECK-LABEL: einsum_no_match_on_invalid_reshape_op_1
// CHECK: "tf.Einsum"
}

func @einsum_no_match_on_invalid_reshape_op_2(%arg0 : tensor<?x36x?xf32>, %arg1 : tensor<?x36x?x32xf32>) -> tensor<?x36x32xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {device = "", equation = "bin,binj->bij"} : (tensor<?x36x?xf32>, tensor<?x36x?x32xf32>) -> tensor<?x36x32xf32>
  return %0 : tensor<?x36x32xf32>
// CHECK-LABEL: einsum_no_match_on_invalid_reshape_op_2
// CHECK: "tf.Einsum"
}

func @einsum_ellipsis(%arg0: tensor<1x512x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<1x512x256xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {device = "", equation = "...x,xy->...y"} : (tensor<1x512x128xf32>, tensor<128x256xf32>) -> tensor<1x512x256xf32>
  return %0 : tensor<1x512x256xf32>
// CHECK-LABEL: einsum_ellipsis
// CHECK: "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x512x128xf32>, tensor<128x256xf32>) -> tensor<1x512x256xf32>
}

func @einsum_ellipsis_in_both_sides(%arg0: tensor<1x11x19xf32>, %arg1: tensor<7x11x13x19xf32>) -> tensor<7x11x13xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {device = "", equation = "...IJ,...INJ->...IN"} : (tensor<1x11x19xf32>, tensor<7x11x13x19xf32>) -> tensor<7x11x13xf32>
  return %0 : tensor<7x11x13xf32>
  // CHECK-LABEL: einsum_ellipsis_in_both_sides
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[1, 11, 1, 19]> : tensor<4xi64>
  // CHECK-DAG: %[[cst_2:.*]] = arith.constant dense<[7, 11, 13]> : tensor<3xi64>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg1, %[[cst]]) : (tensor<7x11x13x19xf32>, tensor<4xi32>) -> tensor<7x11x19x13xf32>
  // CHECK: %[[v1:.*]] = "tf.Reshape"(%arg0, %[[cst_1]]) : (tensor<1x11x19xf32>, tensor<4xi64>) -> tensor<1x11x1x19xf32>
  // CHECK: %[[v2:.*]] = "tf.BatchMatMulV2"(%[[v1]], %[[v0]]) {adj_x = false, adj_y = false} : (tensor<1x11x1x19xf32>, tensor<7x11x19x13xf32>) -> tensor<7x11x1x13xf32>
  // CHECK: %[[v3:.*]] = "tf.Reshape"(%[[v2]], %[[cst_2]]) : (tensor<7x11x1x13xf32>, tensor<3xi64>) -> tensor<7x11x13xf32>
  // CHECK: return %[[v3]] : tensor<7x11x13xf32>
}

func @einsum_ellipsis_with_broadcast(%arg0: tensor<5x4x3xf32>, %arg1: tensor<3x2x1xf32>) -> tensor<4x2x5xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {device = "", equation = "...ij,j...->i..."} : (tensor<5x4x3xf32>, tensor<3x2x1xf32>) -> tensor<4x2x5xf32>
  return %0 : tensor<4x2x5xf32>
  // CHECK-LABEL: einsum_ellipsis_with_broadcast
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
  // CHECK: %[[v0:.*]] = "tf.Transpose"(%arg1, %[[cst]]) : (tensor<3x2x1xf32>, tensor<3xi32>) -> tensor<1x3x2xf32>
  // CHECK: %[[v1:.*]] = "tf.BatchMatMulV2"(%arg0, %[[v0]]) {adj_x = false, adj_y = false} : (tensor<5x4x3xf32>, tensor<1x3x2xf32>) -> tensor<5x4x2xf32>
  // CHECK: %[[v2:.*]] = "tf.Transpose"(%[[v1]], %[[cst_1]]) : (tensor<5x4x2xf32>, tensor<3xi32>) -> tensor<4x2x5xf32>
  // CHECK: return %[[v2]] : tensor<4x2x5xf32>
}
