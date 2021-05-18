// RUN: tf-opt -split-input-file -verify-diagnostics -tf-unroll-batch-matmul %s | FileCheck %s

func @batchMatMulV2TwoDim(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32>
  return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulV2TwoDim
  // CHECK: %[[cst:.*]] = "tf.Const"() {value = dense<[6, 4, 5]> : tensor<3xi64>}
  // CHECK: %[[cst_0:.*]] = "tf.Const"() {value = dense<[1, 4, 5]> : tensor<3xi64>}
  // CHECK: %[[cst_1:.*]] = "tf.Const"() {value = dense<[4, 5]> : tensor<2xi64>}
  // CHECK: %[[cst_2:.*]] = "tf.Const"() {value = dense<[6, 5, 6]> : tensor<3xi64>}
  // CHECK: %[[cst_3:.*]] = "tf.Const"() {value = dense<0> : tensor<3xi64>}
  // CHECK: %[[cst_4:.*]] = "tf.Const"() {value = dense<[1, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_5:.*]] = "tf.Const"() {value = dense<[2, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_6:.*]] = "tf.Const"() {value = dense<[3, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_7:.*]] = "tf.Const"() {value = dense<[4, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_8:.*]] = "tf.Const"() {value = dense<[5, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_9:.*]] = "tf.Const"() {value = dense<[1, 5, 6]> : tensor<3xi64>}
  // CHECK: %[[cst_10:.*]] = "tf.Const"() {value = dense<[5, 6]> : tensor<2xi64>}
  // CHECK: %[[cst_11:.*]] = "tf.Const"() {value = dense<[2, 3, 4, 6]> : tensor<4xi64>}

  // CHECK: %[[v0:.*]] = "tf.Reshape"(%arg0, %[[cst]]) : (tensor<2x3x4x5xf32>, tensor<3xi64>) -> tensor<6x4x5xf32>
  // CHECK: %[[v1:.*]] = "tf.Slice"(%[[v0]], %[[cst_3]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v2:.*]] = "tf.Reshape"(%[[v1]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v3:.*]] = "tf.Slice"(%[[v0]], %[[cst_4]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v4:.*]] = "tf.Reshape"(%[[v3]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v5:.*]] = "tf.Slice"(%[[v0]], %[[cst_5]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v6:.*]] = "tf.Reshape"(%[[v5]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v7:.*]] = "tf.Slice"(%[[v0]], %[[cst_6]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v8:.*]] = "tf.Reshape"(%[[v7]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v9:.*]] = "tf.Slice"(%[[v0]], %[[cst_7]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v10:.*]] = "tf.Reshape"(%[[v9]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v11:.*]] = "tf.Slice"(%[[v0]], %[[cst_8]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v12:.*]] = "tf.Reshape"(%[[v11]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>

  // CHECK: %[[v13:.*]] = "tf.Reshape"(%arg1, %[[cst_2]]) : (tensor<2x3x5x6xf32>, tensor<3xi64>) -> tensor<6x5x6xf32>
  // CHECK: %[[v14:.*]] = "tf.Slice"(%[[v13]], %[[cst_3]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v15:.*]] = "tf.Reshape"(%[[v14]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v16:.*]] = "tf.Slice"(%[[v13]], %[[cst_4]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v17:.*]] = "tf.Reshape"(%[[v16]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v18:.*]] = "tf.Slice"(%[[v13]], %[[cst_5]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v19:.*]] = "tf.Reshape"(%[[v18]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v20:.*]] = "tf.Slice"(%[[v13]], %[[cst_6]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v21:.*]] = "tf.Reshape"(%[[v20]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v22:.*]] = "tf.Slice"(%[[v13]], %[[cst_7]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v23:.*]] = "tf.Reshape"(%[[v22]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v24:.*]] = "tf.Slice"(%[[v13]], %[[cst_8]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v25:.*]] = "tf.Reshape"(%[[v24]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>

  // CHECK: %[[v26:.*]] = "tf.MatMul"(%[[v2]], %[[v15]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v27:.*]] = "tf.MatMul"(%[[v4]], %[[v17]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v28:.*]] = "tf.MatMul"(%[[v6]], %[[v19]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v29:.*]] = "tf.MatMul"(%[[v8]], %[[v21]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v30:.*]] = "tf.MatMul"(%[[v10]], %[[v23]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v31:.*]] = "tf.MatMul"(%[[v12]], %[[v25]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[v32:.*]] = "tf.Pack"(%[[v26]], %[[v27]], %[[v28]], %[[v29]], %[[v30]], %[[v31]]) {axis = 0 : i64} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[v33:.*]] = "tf.Reshape"(%[[v32]], %[[cst_11]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>

  // CHECK: return %[[v33]] : tensor<2x3x4x6xf32>
}

// -----

func @batchMatMulV2FlatInput(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulV2FlatInput
  // CHECK: %[[cst_0:.*]] = "tf.Const"() {value = dense<[1, 4, 5]> : tensor<3xi64>}
  // CHECK: %[[cst_1:.*]] = "tf.Const"() {value = dense<[4, 5]> : tensor<2xi64>}
  // CHECK: %[[cst_2:.*]] = "tf.Const"() {value = dense<0> : tensor<3xi64>}
  // CHECK: %[[cst_3:.*]] = "tf.Const"() {value = dense<[1, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_4:.*]] = "tf.Const"() {value = dense<[2, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_5:.*]] = "tf.Const"() {value = dense<[1, 5, 6]> : tensor<3xi64>}
  // CHECK: %[[cst_6:.*]] = "tf.Const"() {value = dense<[5, 6]> : tensor<2xi64>}

  // CHECK: %[[v0:.*]] = "tf.Slice"(%arg0, %[[cst_2]], %[[cst_0]]) : (tensor<3x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v1:.*]] = "tf.Reshape"(%[[v0]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v2:.*]] = "tf.Slice"(%arg0, %[[cst_3]], %[[cst_0]]) : (tensor<3x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v3:.*]] = "tf.Reshape"(%[[v2]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v4:.*]] = "tf.Slice"(%arg0, %[[cst_4]], %[[cst_0]]) : (tensor<3x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v5:.*]] = "tf.Reshape"(%[[v4]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>

  // CHECK: %[[v6:.*]] = "tf.Slice"(%arg1, %[[cst_2]], %[[cst_5]]) : (tensor<3x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v7:.*]] = "tf.Reshape"(%[[v6]], %[[cst_6]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v8:.*]] = "tf.Slice"(%arg1, %[[cst_3]], %[[cst_5]]) : (tensor<3x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v9:.*]] = "tf.Reshape"(%[[v8]], %[[cst_6]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v10:.*]] = "tf.Slice"(%arg1, %[[cst_4]], %[[cst_5]]) : (tensor<3x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v11:.*]] = "tf.Reshape"(%[[v10]], %[[cst_6]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>

  // CHECK: %[[mm0:.*]] = "tf.MatMul"(%[[v1]], %[[v7]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[mm1:.*]] = "tf.MatMul"(%[[v3]], %[[v9]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[mm2:.*]] = "tf.MatMul"(%[[v5]], %[[v11]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[v17:.*]] = "tf.Pack"(%[[mm0]], %[[mm1]], %[[mm2]]) {axis = 0 : i64} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>

  // CHECK: return %[[v17]] : tensor<3x4x6xf32>
}

// -----

func @batchMatMulV2Matrix(%arg0: tensor<4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>

  // CHECK-LABEL: batchMatMulV2Matrix
  // CHECK: %[[v0:.*]] = "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: return %[[v0]] : tensor<4x6xf32>
}

// -----

func @batchMatMulV3Matrix(%arg0: tensor<4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>

  // CHECK-LABEL: batchMatMulV3Matrix
  // CHECK: %[[v0:.*]] = "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: return %[[v0]] : tensor<4x6xf32>
}

// -----

func @batchMatMulV3MatrixInt8(%arg0: tensor<4x5xi8>, %arg1: tensor<5x6xi8>) -> tensor<4x6xi32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<4x5xi8>, tensor<5x6xi8>) -> tensor<4x6xi32>
  return %0 : tensor<4x6xi32>

  // CHECK-LABEL: batchMatMulV3MatrixInt8
  // CHECK: %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<4x5xi8>, tensor<5x6xi8>) -> tensor<4x6xi32>
  // CHECK: return %0 : tensor<4x6xi32>
}

// -----

func @batchMatMulTwoDim(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32>
  return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulTwoDim
  // CHECK: %[[cst:.*]] = "tf.Const"() {value = dense<[6, 4, 5]> : tensor<3xi64>}
  // CHECK: %[[cst_0:.*]] = "tf.Const"() {value = dense<[1, 4, 5]> : tensor<3xi64>}
  // CHECK: %[[cst_1:.*]] = "tf.Const"() {value = dense<[4, 5]> : tensor<2xi64>}
  // CHECK: %[[cst_2:.*]] = "tf.Const"() {value = dense<[6, 5, 6]> : tensor<3xi64>}
  // CHECK: %[[cst_3:.*]] = "tf.Const"() {value = dense<0> : tensor<3xi64>}
  // CHECK: %[[cst_4:.*]] = "tf.Const"() {value = dense<[1, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_5:.*]] = "tf.Const"() {value = dense<[2, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_6:.*]] = "tf.Const"() {value = dense<[3, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_7:.*]] = "tf.Const"() {value = dense<[4, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_8:.*]] = "tf.Const"() {value = dense<[5, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_9:.*]] = "tf.Const"() {value = dense<[1, 5, 6]> : tensor<3xi64>}
  // CHECK: %[[cst_10:.*]] = "tf.Const"() {value = dense<[5, 6]> : tensor<2xi64>}
  // CHECK: %[[cst_11:.*]] = "tf.Const"() {value = dense<[2, 3, 4, 6]> : tensor<4xi64>}

  // CHECK: %[[v0:.*]] = "tf.Reshape"(%arg0, %[[cst]]) : (tensor<2x3x4x5xf32>, tensor<3xi64>) -> tensor<6x4x5xf32>
  // CHECK: %[[v1:.*]] = "tf.Slice"(%[[v0]], %[[cst_3]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v2:.*]] = "tf.Reshape"(%[[v1]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v3:.*]] = "tf.Slice"(%[[v0]], %[[cst_4]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v4:.*]] = "tf.Reshape"(%[[v3]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v5:.*]] = "tf.Slice"(%[[v0]], %[[cst_5]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v6:.*]] = "tf.Reshape"(%[[v5]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v7:.*]] = "tf.Slice"(%[[v0]], %[[cst_6]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v8:.*]] = "tf.Reshape"(%[[v7]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v9:.*]] = "tf.Slice"(%[[v0]], %[[cst_7]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v10:.*]] = "tf.Reshape"(%[[v9]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v11:.*]] = "tf.Slice"(%[[v0]], %[[cst_8]], %[[cst_0]]) : (tensor<6x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v12:.*]] = "tf.Reshape"(%[[v11]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>

  // CHECK: %[[v13:.*]] = "tf.Reshape"(%arg1, %[[cst_2]]) : (tensor<2x3x5x6xf32>, tensor<3xi64>) -> tensor<6x5x6xf32>
  // CHECK: %[[v14:.*]] = "tf.Slice"(%[[v13]], %[[cst_3]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v15:.*]] = "tf.Reshape"(%[[v14]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v16:.*]] = "tf.Slice"(%[[v13]], %[[cst_4]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v17:.*]] = "tf.Reshape"(%[[v16]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v18:.*]] = "tf.Slice"(%[[v13]], %[[cst_5]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v19:.*]] = "tf.Reshape"(%[[v18]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v20:.*]] = "tf.Slice"(%[[v13]], %[[cst_6]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v21:.*]] = "tf.Reshape"(%[[v20]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v22:.*]] = "tf.Slice"(%[[v13]], %[[cst_7]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v23:.*]] = "tf.Reshape"(%[[v22]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v24:.*]] = "tf.Slice"(%[[v13]], %[[cst_8]], %[[cst_9]]) : (tensor<6x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v25:.*]] = "tf.Reshape"(%[[v24]], %[[cst_10]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>

  // CHECK: %[[v26:.*]] = "tf.MatMul"(%[[v2]], %[[v15]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v27:.*]] = "tf.MatMul"(%[[v4]], %[[v17]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v28:.*]] = "tf.MatMul"(%[[v6]], %[[v19]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v29:.*]] = "tf.MatMul"(%[[v8]], %[[v21]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v30:.*]] = "tf.MatMul"(%[[v10]], %[[v23]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[v31:.*]] = "tf.MatMul"(%[[v12]], %[[v25]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[v32:.*]] = "tf.Pack"(%[[v26]], %[[v27]], %[[v28]], %[[v29]], %[[v30]], %[[v31]]) {axis = 0 : i64} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[v33:.*]] = "tf.Reshape"(%[[v32]], %[[cst_11]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>

  // CHECK: return %[[v33]] : tensor<2x3x4x6xf32>
}

// -----

func @batchMatMulFlatInput(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulFlatInput
  // CHECK: %[[cst_0:.*]] = "tf.Const"() {value = dense<[1, 4, 5]> : tensor<3xi64>}
  // CHECK: %[[cst_1:.*]] = "tf.Const"() {value = dense<[4, 5]> : tensor<2xi64>}
  // CHECK: %[[cst_2:.*]] = "tf.Const"() {value = dense<0> : tensor<3xi64>}
  // CHECK: %[[cst_3:.*]] = "tf.Const"() {value = dense<[1, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_4:.*]] = "tf.Const"() {value = dense<[2, 0, 0]> : tensor<3xi64>}
  // CHECK: %[[cst_5:.*]] = "tf.Const"() {value = dense<[1, 5, 6]> : tensor<3xi64>}
  // CHECK: %[[cst_6:.*]] = "tf.Const"() {value = dense<[5, 6]> : tensor<2xi64>}

  // CHECK: %[[v0:.*]] = "tf.Slice"(%arg0, %[[cst_2]], %[[cst_0]]) : (tensor<3x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v1:.*]] = "tf.Reshape"(%[[v0]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v2:.*]] = "tf.Slice"(%arg0, %[[cst_3]], %[[cst_0]]) : (tensor<3x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v3:.*]] = "tf.Reshape"(%[[v2]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>
  // CHECK: %[[v4:.*]] = "tf.Slice"(%arg0, %[[cst_4]], %[[cst_0]]) : (tensor<3x4x5xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x4x5xf32>
  // CHECK: %[[v5:.*]] = "tf.Reshape"(%[[v4]], %[[cst_1]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>

  // CHECK: %[[v6:.*]] = "tf.Slice"(%arg1, %[[cst_2]], %[[cst_5]]) : (tensor<3x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v7:.*]] = "tf.Reshape"(%[[v6]], %[[cst_6]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v8:.*]] = "tf.Slice"(%arg1, %[[cst_3]], %[[cst_5]]) : (tensor<3x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v9:.*]] = "tf.Reshape"(%[[v8]], %[[cst_6]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>
  // CHECK: %[[v10:.*]] = "tf.Slice"(%arg1, %[[cst_4]], %[[cst_5]]) : (tensor<3x5x6xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x5x6xf32>
  // CHECK: %[[v11:.*]] = "tf.Reshape"(%[[v10]], %[[cst_6]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>

  // CHECK: %[[mm0:.*]] = "tf.MatMul"(%[[v1]], %[[v7]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[mm1:.*]] = "tf.MatMul"(%[[v3]], %[[v9]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[mm2:.*]] = "tf.MatMul"(%[[v5]], %[[v11]]) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[v17:.*]] = "tf.Pack"(%[[mm0]], %[[mm1]], %[[mm2]]) {axis = 0 : i64} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>

  // CHECK: return %[[v17]] : tensor<3x4x6xf32>
}

// -----

func @batchMatMulMatrix(%arg0: tensor<4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>

  // CHECK-LABEL: batchMatMulMatrix
  // CHECK: %[[v0:.*]] = "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: return %[[v0]] : tensor<4x6xf32>
}
