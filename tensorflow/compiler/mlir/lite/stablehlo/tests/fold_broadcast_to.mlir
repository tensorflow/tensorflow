// RUN: odml-to-stablehlo-opt %s -fold-broadcast-to-pass -cse -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @broadcast_mul0
func.func @broadcast_mul0(%arg0: tensor<5x7xf32>, %arg1: tensor<7xf32>) -> tensor<5x7xf32> {
  %cst = mhlo.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg1, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  func.return %1 : tensor<5x7xf32>
  // CHECK:  %0 = tfl.mul(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<5x7xf32>, tensor<7xf32>) -> tensor<5x7xf32>
}

// CHECK-LABEL: @broadcast_mul1
func.func @broadcast_mul1(%arg0: tensor<7xf32>, %arg1: tensor<5x7xf32>) -> tensor<5x7xf32> {
  %cst = mhlo.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tfl.mul"(%0, %arg1) {fused_activation_function = "NONE"} : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  func.return %1 : tensor<5x7xf32>
  // CHECK: %0 = tfl.mul(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
}

// CHECK-LABEL: @broadcast_eq
func.func @broadcast_eq(%arg0: tensor<5x7xf32>, %arg1: tensor<7xf32>) -> tensor<5x7xf32> {
  %cst = mhlo.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg1, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tfl.equal"(%arg0, %0) : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  func.return %1 : tensor<5x7xf32>
  // CHECK: %0 = "tfl.equal"(%arg0, %arg1) : (tensor<5x7xf32>, tensor<7xf32>) -> tensor<5x7xf32>
}

// CHECK-LABEL: @broadcast_eq_no_fold
func.func @broadcast_eq_no_fold(%arg0: tensor<1x2x3x5x7xf32>, %arg1: tensor<7xf32>) -> tensor<1x2x3x5x7xf32> {
  %cst = mhlo.constant dense<[1, 2, 3, 5, 7]> : tensor<5xi32>
  %0 = "tfl.broadcast_to"(%arg1, %cst) : (tensor<7xf32>, tensor<5xi32>) -> tensor<1x2x3x5x7xf32>
  %1 = "tfl.equal"(%arg0, %0) : (tensor<1x2x3x5x7xf32>, tensor<1x2x3x5x7xf32>) -> tensor<1x2x3x5x7xf32>
  func.return %1 : tensor<1x2x3x5x7xf32>
  // CHECK: %2 = "tfl.equal"(%arg0, %1) : (tensor<1x2x3x5x7xf32>, tensor<1x2x3x5x7xf32>) -> tensor<1x2x3x5x7xf32>
}

// CHECK-LABEL: @broadcast_batchmatmul
func.func @broadcast_batchmatmul(%arg0: tensor<5x30x1024xf32>) -> tensor<5x30x8192xf32> {
  %cst = arith.constant dense_resource<__elided__> : tensor<1024x8192xf32>
  %cst_1 = mhlo.constant dense<[5, 1024, 8192]> : tensor<3xi32>
  %0 = "tfl.broadcast_to"(%cst, %cst_1) : (tensor<1024x8192xf32>, tensor<3xi32>) -> tensor<5x1024x8192xf32>
  %1 = "tfl.batch_matmul"(%arg0, %0) {adj_x = false, adj_y = false} : (tensor<5x30x1024xf32>, tensor<5x1024x8192xf32>) -> tensor<5x30x8192xf32>
  return %1 : tensor<5x30x8192xf32>
  // CHECK: %0 = "tfl.batch_matmul"(%arg0, %cst) <{adj_x = false, adj_y = false}> : (tensor<5x30x1024xf32>, tensor<1024x8192xf32>) -> tensor<5x30x8192xf32>
}
