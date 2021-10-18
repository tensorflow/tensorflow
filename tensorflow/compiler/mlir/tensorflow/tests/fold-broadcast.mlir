// RUN: tf-opt -tf-broadcast-fold %s | FileCheck %s

// CHECK-LABEL: @broadcast_mul0
func @broadcast_mul0(%arg0: tensor<5x7xf32>, %arg1: tensor<7xf32>) -> tensor<5x7xf32> {
  %cst = arith.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tf.BroadcastTo"(%arg1, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tf.Mul"(%arg0, %0) : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  return %1 : tensor<5x7xf32>
  // CHECK: %[[V0:.*]] = "tf.Mul"(%arg0, %arg1) : (tensor<5x7xf32>, tensor<7xf32>) -> tensor<5x7xf32>
  // CHECK: %[[V0]] : tensor<5x7xf32>
}

// CHECK-LABEL: @broadcast_mul1
func @broadcast_mul1(%arg0: tensor<7xf32>, %arg1: tensor<5x7xf32>) -> tensor<5x7xf32> {
  %cst = arith.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tf.Mul"(%0, %arg1) : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  return %1 : tensor<5x7xf32>
  // CHECK: %[[V0:.*]] = "tf.Mul"(%arg0, %arg1) : (tensor<7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  // CHECK: %[[V0]] : tensor<5x7xf32>
}

// CHECK-LABEL: @broadcast_add_implicit_fold
func @broadcast_add_implicit_fold(%arg0: tensor<5x1xf32>, %arg1: tensor<7xf32>) -> tensor<5x7xf32> {
  %cst = arith.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tf.BroadcastTo"(%arg1, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tf.AddV2"(%arg0, %0) : (tensor<5x1xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  return %1 : tensor<5x7xf32>
  // CHECK: %[[V0:.*]] = "tf.AddV2"(%arg0, %arg1) : (tensor<5x1xf32>, tensor<7xf32>) -> tensor<5x7xf32>
  // CHECK: %[[V0]] : tensor<5x7xf32>
}

// CHECK-LABEL: @broadcast_mul_implicit_no_fold
func @broadcast_mul_implicit_no_fold(%arg0: tensor<5x7xf32>, %arg1: tensor<5xf32>) -> tensor<3x5x7xf32> {
  %cst = arith.constant dense<[3, 5, 7]> : tensor<3xi32>
  %0 = "tf.BroadcastTo"(%arg1, %cst) : (tensor<5xf32>, tensor<3xi32>) -> tensor<3x5x7xf32>
  %1 = "tf.Mul"(%arg0, %0) : (tensor<5x7xf32>, tensor<3x5x7xf32>) -> tensor<3x5x7xf32>
  return %1 : tensor<3x5x7xf32>
  // CHECK: %[[C0:.*]] = arith.constant dense<[3, 5, 7]> : tensor<3xi32>
  // CHECK: %[[V0:.*]] = "tf.BroadcastTo"(%arg1, %[[C0]]) : (tensor<5xf32>, tensor<3xi32>) -> tensor<3x5x7xf32>
  // CHECK: %[[V1:.*]] = "tf.Mul"(%arg0, %[[V0]]) : (tensor<5x7xf32>, tensor<3x5x7xf32>) -> tensor<3x5x7xf32>
  // CHECK: %[[V1]] : tensor<3x5x7xf32>
}

// CHECK-LABEL: @broadcast_eq
func @broadcast_eq(%arg0: tensor<5x7xf32>, %arg1: tensor<7xf32>) -> tensor<5x7xi1> {
  %cst = arith.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tf.BroadcastTo"(%arg1, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tf.Equal"(%arg0, %0) {incompatible_shape_error = true} : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xi1>
  return %1 : tensor<5x7xi1>
  // CHECK: %[[V0:.*]] = "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = true} : (tensor<5x7xf32>, tensor<7xf32>) -> tensor<5x7xi1>
  // CHECK: %[[V0]] : tensor<5x7xi1>
}

// CHECK-LABEL: @broadcast_neq
func @broadcast_neq(%arg0: tensor<5x7xf32>, %arg1: tensor<7xf32>) -> tensor<5x7xi1> {
  %cst = arith.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tf.BroadcastTo"(%arg1, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tf.NotEqual"(%arg0, %0) {incompatible_shape_error = true} : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xi1>
  return %1 : tensor<5x7xi1>
  // CHECK: %[[V0:.*]] = "tf.NotEqual"(%arg0, %arg1) {incompatible_shape_error = true} : (tensor<5x7xf32>, tensor<7xf32>) -> tensor<5x7xi1>
  // CHECK: %[[V0]] : tensor<5x7xi1>
}

// CHECK-LABEL: @broadcast_both_operand
func @broadcast_both_operand(%arg0: tensor<7xf32>, %arg1: tensor<5x1xf32>) -> tensor<5x7xf32> {
  %cst = arith.constant dense<[5, 7]> : tensor<2xi64>
  %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<7xf32>, tensor<2xi64>) -> tensor<5x7xf32>
  %1 = "tf.BroadcastTo"(%arg1, %cst) : (tensor<5x1xf32>, tensor<2xi64>) -> tensor<5x7xf32>
  %2 = "tf.Add"(%0, %1) : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  return %2 : tensor<5x7xf32>
  // CHECK: %[[V0:.*]] = "tf.Add"(%arg0, %arg1) : (tensor<7xf32>, tensor<5x1xf32>) -> tensor<5x7xf32>
  // CHECK: %[[V0]] : tensor<5x7xf32>
}

// CHECK-LABEL: @broadcast_batch_matmul_v2_rhs
func @broadcast_batch_matmul_v2_rhs(%arg0: tensor<17x17x17xf32>, %arg1: tensor<17x24xf32>) -> tensor<17x17x24xf32> {
  %cst = arith.constant dense<[17, 17, 24]> : tensor<3xi64>
  %0 = "tf.BroadcastTo"(%arg1, %cst) : (tensor<17x24xf32>, tensor<3xi64>) -> tensor<17x17x24xf32>
  %1 = "tf.BatchMatMulV2"(%arg0, %0) {adj_x = false, adj_y = false} : (tensor<17x17x17xf32>, tensor<17x17x24xf32>) -> tensor<17x17x24xf32>
  return %1 : tensor<17x17x24xf32>
  // CHECK: %[[V0:.*]] = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<17x17x17xf32>, tensor<17x24xf32>) -> tensor<17x17x24xf32>
  // CHECK: %[[V0]] : tensor<17x17x24xf32>
}

// CHECK-LABEL: @broadcast_batch_matmul_v2_lhs
func @broadcast_batch_matmul_v2_lhs(%arg0: tensor<17x17xf32>, %arg1: tensor<17x17x24xf32>) -> tensor<17x17x24xf32> {
  %cst = arith.constant dense<[17, 17, 17]> : tensor<3xi64>
  %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<17x17xf32>, tensor<3xi64>) -> tensor<17x17x17xf32>
  %1 = "tf.BatchMatMulV2"(%0, %arg1) {adj_x = false, adj_y = false} : (tensor<17x17x17xf32>, tensor<17x17x24xf32>) -> tensor<17x17x24xf32>
  return %1 : tensor<17x17x24xf32>
  // CHECK: %[[V0:.*]] = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<17x17xf32>, tensor<17x17x24xf32>) -> tensor<17x17x24xf32>
  // CHECK: %[[V0]] : tensor<17x17x24xf32>
}

// CHECK-LABEL: @broadcast_batch_matmul_v2_failed
func @broadcast_batch_matmul_v2_failed(%arg0: tensor<17x17x1xf32>, %arg1: tensor<17x17x24xf32>) -> tensor<17x17x24xf32> {
  %cst = arith.constant dense<[17, 17, 17]> : tensor<3xi64>
  %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<17x17x1xf32>, tensor<3xi64>) -> tensor<17x17x17xf32>
  %1 = "tf.BatchMatMulV2"(%0, %arg1) {adj_x = false, adj_y = false} : (tensor<17x17x17xf32>, tensor<17x17x24xf32>) -> tensor<17x17x24xf32>
  return %1 : tensor<17x17x24xf32>
  // CHECK: %[[V0:.*]] = "tf.BroadcastTo"
  // CHECK: "tf.BatchMatMulV2"(%[[V0]], %arg1)
}

// CHECK-LABEL: @broadcast_splat_operand
func @broadcast_splat_operand() -> tensor<5x5xi64> {
  %cst = arith.constant dense<5> : tensor<2xi64>
  %0 = "tf.BroadcastTo"(%cst, %cst) : (tensor<2xi64>, tensor<2xi64>) -> tensor<5x5xi64>
  return %0 : tensor<5x5xi64>
  // CHECK: %[[V0:.*]] = "tf.Const"() {value = dense<5> : tensor<5x5xi64>} : () -> tensor<5x5xi64>
  // CHECK: %[[V0]] : tensor<5x5xi64>
}
