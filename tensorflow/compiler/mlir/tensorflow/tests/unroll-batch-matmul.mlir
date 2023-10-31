// RUN: tf-opt -split-input-file -verify-diagnostics -tf-unroll-batch-matmul %s | FileCheck %s

//==== V1 tests ====

func.func @batchMatMulTwoDim(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulTwoDim
  // CHECK-DAG: %[[LHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 4, 5]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 5, 6]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RESULT_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 3, 4, 6]> : tensor<4xi64>}>

  // CHECK: %[[LHS_RESHAPED:.*]] = "tf.Reshape"(%arg0, %[[LHS_RESHAPED_SHAPE]]) : (tensor<2x3x4x5xf32>, tensor<3xi64>) -> tensor<6x4x5xf32>
  // CHECK: %[[LHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[LHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)

  // CHECK: %[[RHS_RESHAPED:.*]] = "tf.Reshape"(%arg1, %[[RHS_RESHAPED_SHAPE]]) : (tensor<2x3x5x6xf32>, tensor<3xi64>) -> tensor<6x5x6xf32>
  // CHECK: %[[RHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[RHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#2, %[[RHS_SPLIT]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_4:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#3, %[[RHS_SPLIT]]#3) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_5:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#4, %[[RHS_SPLIT]]#4) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_6:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#5, %[[RHS_SPLIT]]#5) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]], %[[MATMUL_4]], %[[MATMUL_5]], %[[MATMUL_6]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Reshape"(%[[MATMUL_PACKED]], %[[RESULT_SHAPE]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>
  // CHECK: return %[[RESULT]] : tensor<2x3x4x6xf32>
}

// -----

func.func @batchMatMulTwoDimAdjXY(%arg0: tensor<2x3x5x4xf32>, %arg1: tensor<2x3x6x5xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = true, adj_y = true} : (tensor<2x3x5x4xf32>, tensor<2x3x6x5xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulTwoDimAdjXY
  // CHECK-DAG: %[[LHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 5, 4]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 6, 5]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RESULT_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 3, 4, 6]> : tensor<4xi64>}>

  // CHECK: %[[LHS_RESHAPED:.*]] = "tf.Reshape"(%arg0, %[[LHS_RESHAPED_SHAPE]]) : (tensor<2x3x5x4xf32>, tensor<3xi64>) -> tensor<6x5x4xf32>
  // CHECK: %[[LHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[LHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x5x4xf32>) -> (tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>)

  // CHECK: %[[RHS_RESHAPED:.*]] = "tf.Reshape"(%arg1, %[[RHS_RESHAPED_SHAPE]]) : (tensor<2x3x6x5xf32>, tensor<3xi64>) -> tensor<6x6x5xf32>
  // CHECK: %[[RHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[RHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x6x5xf32>) -> (tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#0) <{grad_a = false, grad_b = false, transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#1) <{grad_a = false, grad_b = false, transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#2, %[[RHS_SPLIT]]#2) <{grad_a = false, grad_b = false, transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_4:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#3, %[[RHS_SPLIT]]#3) <{grad_a = false, grad_b = false, transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_5:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#4, %[[RHS_SPLIT]]#4) <{grad_a = false, grad_b = false, transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_6:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#5, %[[RHS_SPLIT]]#5) <{grad_a = false, grad_b = false, transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]], %[[MATMUL_4]], %[[MATMUL_5]], %[[MATMUL_6]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Reshape"(%[[MATMUL_PACKED]], %[[RESULT_SHAPE]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>
  // CHECK: return %[[RESULT]] : tensor<2x3x4x6xf32>
}

// -----

func.func @batchMatMulOneDim(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulOneDim
  // CHECK: %[[LHS_RESHAPED:.*]]:3 = "tf.Unpack"(%arg0) <{axis = 0 : i64}> : (tensor<3x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)
  // CHECK: %[[RHS_RESHAPED:.*]]:3 = "tf.Unpack"(%arg1) <{axis = 0 : i64}> : (tensor<3x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_RESHAPED]]#0, %[[RHS_RESHAPED]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_RESHAPED]]#1, %[[RHS_RESHAPED]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_RESHAPED]]#2, %[[RHS_RESHAPED]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<3x4x6xf32>
}

// -----

func.func @batchMatMulSingleBatch(%arg0: tensor<1x4x5xf32>, %arg1: tensor<1x5x6xf32>) -> tensor<1x4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) : (tensor<1x4x5xf32>, tensor<1x5x6xf32>) -> tensor<1x4x6xf32>
  func.return %0 : tensor<1x4x6xf32>

  // CHECK-LABEL: batchMatMulSingleBatch
  // CHECK-DAG: %[[MATMUL_LHS_SHAPE:.*]] = "tf.Const"() <{value = dense<[4, 5]> : tensor<2xi64>}> : () -> tensor<2xi64>
  // CHECK-DAG: %[[MATMUL_RHS_SHAPE:.*]] = "tf.Const"() <{value = dense<[5, 6]> : tensor<2xi64>}> : () -> tensor<2xi64>

  // CHECK: %[[LHS_1:.*]] = "tf.Reshape"(%arg0, %[[MATMUL_LHS_SHAPE]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>

  // CHECK: %[[RHS_2:.*]] = "tf.Reshape"(%arg1, %[[MATMUL_RHS_SHAPE]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_1]], %[[RHS_2]]) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]]) <{axis = 0 : i64}> : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<1x4x6xf32>
}

// -----

func.func @batchMatMulUnbatchedLeft(%arg0: tensor<4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) : (tensor<4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulUnbatchedLeft
  // CHECK: %[[RHS_RESHAPED:.*]]:3 = "tf.Unpack"(%arg1) <{axis = 0 : i64}> : (tensor<3x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %[[RHS_RESHAPED]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%arg0, %[[RHS_RESHAPED]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%arg0, %[[RHS_RESHAPED]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<3x4x6xf32>
}

// -----

func.func @batchMatMulUnbatchedRight(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulUnbatchedRight
  // CHECK: %[[LHS_SPLIT:.*]]:3 = "tf.Unpack"(%arg0) <{axis = 0 : i64}> : (tensor<3x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#2, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<3x4x6xf32>
}

// -----

func.func @batchMatMulMatrix(%arg0: tensor<4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  func.return %0 : tensor<4x6xf32>

  // CHECK-LABEL: batchMatMulMatrix
  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: return %[[MATMUL_1]] : tensor<4x6xf32>
}

// -----

func.func @batchMatMulMatrixAdjXY(%arg0: tensor<5x4xf32>, %arg1: tensor<6x5xf32>) -> tensor<4x6xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = true, adj_y = true} : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  func.return %0 : tensor<4x6xf32>

  // CHECK-LABEL: batchMatMulMatrixAdjXY
  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: return %[[MATMUL_1]] : tensor<4x6xf32>
}

// -----
// ==== V2 tests ====

func.func @batchMatMulV2TwoDim(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulV2TwoDim
  // CHECK-DAG: %[[LHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 4, 5]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 5, 6]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RESULT_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 3, 4, 6]> : tensor<4xi64>}>

  // CHECK: %[[LHS_RESHAPED:.*]] = "tf.Reshape"(%arg0, %[[LHS_RESHAPED_SHAPE]]) : (tensor<2x3x4x5xf32>, tensor<3xi64>) -> tensor<6x4x5xf32>
  // CHECK: %[[LHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[LHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)

  // CHECK: %[[RHS_RESHAPED:.*]] = "tf.Reshape"(%arg1, %[[RHS_RESHAPED_SHAPE]]) : (tensor<2x3x5x6xf32>, tensor<3xi64>) -> tensor<6x5x6xf32>
  // CHECK: %[[RHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[RHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#2, %[[RHS_SPLIT]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_4:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#3, %[[RHS_SPLIT]]#3) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_5:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#4, %[[RHS_SPLIT]]#4) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_6:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#5, %[[RHS_SPLIT]]#5) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]], %[[MATMUL_4]], %[[MATMUL_5]], %[[MATMUL_6]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Reshape"(%[[MATMUL_PACKED]], %[[RESULT_SHAPE]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>
  // CHECK: return %[[RESULT]] : tensor<2x3x4x6xf32>
}

// -----

func.func @batchMatMulV2TwoDimAdjXY(%arg0: tensor<2x3x5x4xf32>, %arg1: tensor<2x3x6x5xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = true} : (tensor<2x3x5x4xf32>, tensor<2x3x6x5xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulV2TwoDimAdjXY
  // CHECK-DAG: %[[LHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 5, 4]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 6, 5]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RESULT_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 3, 4, 6]> : tensor<4xi64>}>

  // CHECK: %[[LHS_RESHAPED:.*]] = "tf.Reshape"(%arg0, %[[LHS_RESHAPED_SHAPE]]) : (tensor<2x3x5x4xf32>, tensor<3xi64>) -> tensor<6x5x4xf32>
  // CHECK: %[[LHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[LHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x5x4xf32>) -> (tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>)

  // CHECK: %[[RHS_RESHAPED:.*]] = "tf.Reshape"(%arg1, %[[RHS_RESHAPED_SHAPE]]) : (tensor<2x3x6x5xf32>, tensor<3xi64>) -> tensor<6x6x5xf32>
  // CHECK: %[[RHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[RHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x6x5xf32>) -> (tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#0) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#1) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#2, %[[RHS_SPLIT]]#2) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_4:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#3, %[[RHS_SPLIT]]#3) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_5:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#4, %[[RHS_SPLIT]]#4) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_6:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#5, %[[RHS_SPLIT]]#5) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]], %[[MATMUL_4]], %[[MATMUL_5]], %[[MATMUL_6]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Reshape"(%[[MATMUL_PACKED]], %[[RESULT_SHAPE]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>
  // CHECK: return %[[RESULT]] : tensor<2x3x4x6xf32>
}

// -----

func.func @batchMatMulV2Broadcast(%arg0: tensor<2x1x4x5xf32>, %arg1: tensor<1x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<2x1x4x5xf32>, tensor<1x3x5x6xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulV2Broadcast
  // CHECK-DAG: %[[LHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 4, 5]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[3, 5, 6]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RESULT_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 3, 4, 6]> : tensor<4xi64>}>

  // CHECK: %[[LHS_RESHAPED:.*]] = "tf.Reshape"(%arg0, %[[LHS_RESHAPED_SHAPE]]) : (tensor<2x1x4x5xf32>, tensor<3xi64>) -> tensor<2x4x5xf32>
  // CHECK: %[[LHS_SPLIT:.*]]:2 = "tf.Unpack"(%[[LHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<2x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>)

  // CHECK: %[[RHS_RESHAPED:.*]] = "tf.Reshape"(%arg1, %[[RHS_RESHAPED_SHAPE]]) : (tensor<1x3x5x6xf32>, tensor<3xi64>) -> tensor<3x5x6xf32>
  // CHECK: %[[RHS_SPLIT:.*]]:3 = "tf.Unpack"(%[[RHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<3x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_4:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_5:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_6:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]], %[[MATMUL_4]], %[[MATMUL_5]], %[[MATMUL_6]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Reshape"(%[[MATMUL_PACKED]], %[[RESULT_SHAPE]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>
  // CHECK: return %[[RESULT]] : tensor<2x3x4x6xf32>
}

// -----

func.func @batchMatMulV2OneDim(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulV2OneDim
  // CHECK: %[[LHS_RESHAPED:.*]]:3 = "tf.Unpack"(%arg0) <{axis = 0 : i64}> : (tensor<3x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)
  // CHECK: %[[RHS_RESHAPED:.*]]:3 = "tf.Unpack"(%arg1) <{axis = 0 : i64}> : (tensor<3x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_RESHAPED]]#0, %[[RHS_RESHAPED]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_RESHAPED]]#1, %[[RHS_RESHAPED]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_RESHAPED]]#2, %[[RHS_RESHAPED]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<3x4x6xf32>
}

// -----

func.func @batchMatMulV2SingleBatch(%arg0: tensor<1x4x5xf32>, %arg1: tensor<1x5x6xf32>) -> tensor<1x4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<1x4x5xf32>, tensor<1x5x6xf32>) -> tensor<1x4x6xf32>
  func.return %0 : tensor<1x4x6xf32>

  // CHECK-LABEL: batchMatMulV2SingleBatch
  // CHECK-DAG: %[[MATMUL_LHS_SHAPE:.*]] = "tf.Const"() <{value = dense<[4, 5]> : tensor<2xi64>}> : () -> tensor<2xi64>
  // CHECK-DAG: %[[MATMUL_RHS_SHAPE:.*]] = "tf.Const"() <{value = dense<[5, 6]> : tensor<2xi64>}> : () -> tensor<2xi64>

  // CHECK: %[[LHS_1:.*]] = "tf.Reshape"(%arg0, %[[MATMUL_LHS_SHAPE]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>

  // CHECK: %[[RHS_2:.*]] = "tf.Reshape"(%arg1, %[[MATMUL_RHS_SHAPE]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_1]], %[[RHS_2]]) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]]) <{axis = 0 : i64}> : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<1x4x6xf32>
}

// -----

func.func @batchMatMulV2UnbatchedLeft(%arg0: tensor<4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulV2UnbatchedLeft
  // CHECK: %[[RHS_RESHAPED:.*]]:3 = "tf.Unpack"(%arg1) <{axis = 0 : i64}> : (tensor<3x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %[[RHS_RESHAPED]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%arg0, %[[RHS_RESHAPED]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%arg0, %[[RHS_RESHAPED]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<3x4x6xf32>
}

// -----

func.func @batchMatMulV2UnbatchedRight(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulV2UnbatchedRight
  // CHECK: %[[LHS_SPLIT:.*]]:3 = "tf.Unpack"(%arg0) <{axis = 0 : i64}> : (tensor<3x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#2, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<3x4x6xf32>
}

// -----

func.func @batchMatMulV2Matrix(%arg0: tensor<4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  func.return %0 : tensor<4x6xf32>

  // CHECK-LABEL: batchMatMulV2Matrix
  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: return %[[MATMUL_1]] : tensor<4x6xf32>
}

// -----

func.func @batchMatMulV2MatrixAdjXY(%arg0: tensor<5x4xf32>, %arg1: tensor<6x5xf32>) -> tensor<4x6xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = true} : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  func.return %0 : tensor<4x6xf32>

  // CHECK-LABEL: batchMatMulV2MatrixAdjXY
  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: return %[[MATMUL_1]] : tensor<4x6xf32>
}

// -----

func.func @batchMatMulV2DynamicSize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<?x?xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  func.return %0 : tensor<?x4xf32>

  // CHECK-LABEL: batchMatMulV2DynamicSize
  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<?x?xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  // CHECK: return %[[MATMUL_1]] : tensor<?x4xf32>
}

// -----
// ==== V3 tests ====

func.func @batchMatMulV3TwoDim(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<2x3x5x6xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulV3TwoDim
  // CHECK-DAG: %[[LHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 4, 5]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 5, 6]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RESULT_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 3, 4, 6]> : tensor<4xi64>}>

  // CHECK: %[[LHS_RESHAPED:.*]] = "tf.Reshape"(%arg0, %[[LHS_RESHAPED_SHAPE]]) : (tensor<2x3x4x5xf32>, tensor<3xi64>) -> tensor<6x4x5xf32>
  // CHECK: %[[LHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[LHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)

  // CHECK: %[[RHS_RESHAPED:.*]] = "tf.Reshape"(%arg1, %[[RHS_RESHAPED_SHAPE]]) : (tensor<2x3x5x6xf32>, tensor<3xi64>) -> tensor<6x5x6xf32>
  // CHECK: %[[RHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[RHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#2, %[[RHS_SPLIT]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_4:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#3, %[[RHS_SPLIT]]#3) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_5:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#4, %[[RHS_SPLIT]]#4) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_6:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#5, %[[RHS_SPLIT]]#5) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]], %[[MATMUL_4]], %[[MATMUL_5]], %[[MATMUL_6]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Reshape"(%[[MATMUL_PACKED]], %[[RESULT_SHAPE]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>
  // CHECK: return %[[RESULT]] : tensor<2x3x4x6xf32>
}

// -----

func.func @batchMatMulV3TwoDimAdjXY(%arg0: tensor<2x3x5x4xf32>, %arg1: tensor<2x3x6x5xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) {adj_x = true, adj_y = true} : (tensor<2x3x5x4xf32>, tensor<2x3x6x5xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulV3TwoDimAdjXY
  // CHECK-DAG: %[[LHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 5, 4]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[6, 6, 5]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RESULT_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 3, 4, 6]> : tensor<4xi64>}>

  // CHECK: %[[LHS_RESHAPED:.*]] = "tf.Reshape"(%arg0, %[[LHS_RESHAPED_SHAPE]]) : (tensor<2x3x5x4xf32>, tensor<3xi64>) -> tensor<6x5x4xf32>
  // CHECK: %[[LHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[LHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x5x4xf32>) -> (tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>)

  // CHECK: %[[RHS_RESHAPED:.*]] = "tf.Reshape"(%arg1, %[[RHS_RESHAPED_SHAPE]]) : (tensor<2x3x6x5xf32>, tensor<3xi64>) -> tensor<6x6x5xf32>
  // CHECK: %[[RHS_SPLIT:.*]]:6 = "tf.Unpack"(%[[RHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<6x6x5xf32>) -> (tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>, tensor<6x5xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#0) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#1) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#2, %[[RHS_SPLIT]]#2) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_4:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#3, %[[RHS_SPLIT]]#3) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_5:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#4, %[[RHS_SPLIT]]#4) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_6:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#5, %[[RHS_SPLIT]]#5) <{transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]], %[[MATMUL_4]], %[[MATMUL_5]], %[[MATMUL_6]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Reshape"(%[[MATMUL_PACKED]], %[[RESULT_SHAPE]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>
  // CHECK: return %[[RESULT]] : tensor<2x3x4x6xf32>
}

// -----

func.func @batchMatMulV3Broadcast(%arg0: tensor<2x1x4x5xf32>, %arg1: tensor<1x3x5x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<2x1x4x5xf32>, tensor<1x3x5x6xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>

  // CHECK-LABEL: batchMatMulV3Broadcast
  // CHECK-DAG: %[[LHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 4, 5]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RHS_RESHAPED_SHAPE:.*]] = "tf.Const"() <{value = dense<[3, 5, 6]> : tensor<3xi64>}>
  // CHECK-DAG: %[[RESULT_SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 3, 4, 6]> : tensor<4xi64>}>

  // CHECK: %[[LHS_RESHAPED:.*]] = "tf.Reshape"(%arg0, %[[LHS_RESHAPED_SHAPE]]) : (tensor<2x1x4x5xf32>, tensor<3xi64>) -> tensor<2x4x5xf32>
  // CHECK: %[[LHS_SPLIT:.*]]:2 = "tf.Unpack"(%[[LHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<2x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>)

  // CHECK: %[[RHS_RESHAPED:.*]] = "tf.Reshape"(%arg1, %[[RHS_RESHAPED_SHAPE]]) : (tensor<1x3x5x6xf32>, tensor<3xi64>) -> tensor<3x5x6xf32>
  // CHECK: %[[RHS_SPLIT:.*]]:3 = "tf.Unpack"(%[[RHS_RESHAPED]]) <{axis = 0 : i64}> : (tensor<3x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %[[RHS_SPLIT]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_4:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_5:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_6:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %[[RHS_SPLIT]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]], %[[MATMUL_4]], %[[MATMUL_5]], %[[MATMUL_6]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<6x4x6xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Reshape"(%[[MATMUL_PACKED]], %[[RESULT_SHAPE]]) : (tensor<6x4x6xf32>, tensor<4xi64>) -> tensor<2x3x4x6xf32>
  // CHECK: return %[[RESULT]] : tensor<2x3x4x6xf32>
}

// -----

func.func @batchMatMulV3OneDim(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulV3OneDim
  // CHECK: %[[LHS_RESHAPED:.*]]:3 = "tf.Unpack"(%arg0) <{axis = 0 : i64}> : (tensor<3x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)
  // CHECK: %[[RHS_RESHAPED:.*]]:3 = "tf.Unpack"(%arg1) <{axis = 0 : i64}> : (tensor<3x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_RESHAPED]]#0, %[[RHS_RESHAPED]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_RESHAPED]]#1, %[[RHS_RESHAPED]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_RESHAPED]]#2, %[[RHS_RESHAPED]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<3x4x6xf32>
}

// -----

func.func @batchMatMulV3SingleBatch(%arg0: tensor<1x4x5xf32>, %arg1: tensor<1x5x6xf32>) -> tensor<1x4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<1x4x5xf32>, tensor<1x5x6xf32>) -> tensor<1x4x6xf32>
  func.return %0 : tensor<1x4x6xf32>

  // CHECK-LABEL: batchMatMulV3SingleBatch
  // CHECK-DAG: %[[MATMUL_LHS_SHAPE:.*]] = "tf.Const"() <{value = dense<[4, 5]> : tensor<2xi64>}> : () -> tensor<2xi64>
  // CHECK-DAG: %[[MATMUL_RHS_SHAPE:.*]] = "tf.Const"() <{value = dense<[5, 6]> : tensor<2xi64>}> : () -> tensor<2xi64>

  // CHECK: %[[LHS_1:.*]] = "tf.Reshape"(%arg0, %[[MATMUL_LHS_SHAPE]]) : (tensor<1x4x5xf32>, tensor<2xi64>) -> tensor<4x5xf32>

  // CHECK: %[[RHS_2:.*]] = "tf.Reshape"(%arg1, %[[MATMUL_RHS_SHAPE]]) : (tensor<1x5x6xf32>, tensor<2xi64>) -> tensor<5x6xf32>

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_1]], %[[RHS_2]]) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]]) <{axis = 0 : i64}> : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<1x4x6xf32>
}

// -----

func.func @batchMatMulV3UnbatchedLeft(%arg0: tensor<4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulV3UnbatchedLeft
  // CHECK: %[[RHS_RESHAPED:.*]]:3 = "tf.Unpack"(%arg1) <{axis = 0 : i64}> : (tensor<3x5x6xf32>) -> (tensor<5x6xf32>, tensor<5x6xf32>, tensor<5x6xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %[[RHS_RESHAPED]]#0) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%arg0, %[[RHS_RESHAPED]]#1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%arg0, %[[RHS_RESHAPED]]#2) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<3x4x6xf32>
}

// -----

func.func @batchMatMulV3UnbatchedRight(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>

  // CHECK-LABEL: batchMatMulV3UnbatchedRight
  // CHECK: %[[LHS_SPLIT:.*]]:3 = "tf.Unpack"(%arg0) <{axis = 0 : i64}> : (tensor<3x4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>, tensor<4x5xf32>)

  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#1, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%[[LHS_SPLIT]]#2, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>

  // CHECK: %[[MATMUL_PACKED:.*]] = "tf.Pack"(%[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]]) <{axis = 0 : i64}> : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<3x4x6xf32>
  // CHECK: return %[[MATMUL_PACKED]] : tensor<3x4x6xf32>
}

// -----

func.func @batchMatMulV3Matrix(%arg0: tensor<4x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  func.return %0 : tensor<4x6xf32>

  // CHECK-LABEL: batchMatMulV3Matrix
  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<4x6xf32>
  // CHECK: return %[[MATMUL_1]] : tensor<4x6xf32>
}

// -----

func.func @batchMatMulV3MatrixAdjXY(%arg0: tensor<5x4xf32>, %arg1: tensor<6x5xf32>) -> tensor<4x6xf32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) {adj_x = true, adj_y = true} : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  func.return %0 : tensor<4x6xf32>

  // CHECK-LABEL: batchMatMulV3MatrixAdjXY
  // CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = true, transpose_b = true}> : (tensor<5x4xf32>, tensor<6x5xf32>) -> tensor<4x6xf32>
  // CHECK: return %[[MATMUL_1]] : tensor<4x6xf32>
}

// -----

func.func @batchMatMulV3MatrixInt8(%arg0: tensor<4x5xi8>, %arg1: tensor<5x6xi8>) -> tensor<4x6xi32> {
  %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<4x5xi8>, tensor<5x6xi8>) -> tensor<4x6xi32>
  func.return %0 : tensor<4x6xi32>

  // CHECK-LABEL: batchMatMulV3MatrixInt8
  // CHECK: %0 = "tf.BatchMatMulV3"(%arg0, %arg1) : (tensor<4x5xi8>, tensor<5x6xi8>) -> tensor<4x6xi32>
  // CHECK: return %0 : tensor<4x6xi32>
}
