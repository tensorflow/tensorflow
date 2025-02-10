// RUN: tf-opt -tfl-optimize-broadcast-like -split-input-file %s | FileCheck %s

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

// CHECK-LABEL: FoldFillOpIntoDivOpRHS
func.func @FoldFillOpIntoDivOpRHS(%arg0: tensor<1x4x1440x1440xf32>, %arg1: tensor<4xi64>) -> tensor<1x4x1440x1440xf32> {
  %cst_2 = arith.constant dense<5.0> : tensor<f32>

  %1 = "tfl.fill"(%arg1, %cst_2) : (tensor<4xi64>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  %36 = tfl.div %arg0, %1 {fused_activation_function = "NONE"} : tensor<1x4x1440x1440xf32>
  return %36 : tensor<1x4x1440x1440xf32>
  // CHECK: %cst = arith.constant dense<5.000000e+00> : tensor<f32>
  // CHECK: %0 = tfl.div(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<1x4x1440x1440xf32>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  // CHECK: return %0 : tensor<1x4x1440x1440xf32>
}

// CHECK-LABEL: FoldFillOpIntoDivOpLHS
func.func @FoldFillOpIntoDivOpLHS(%arg0: tensor<1x4x1440x1440xf32>, %arg1: tensor<4xi64>) -> tensor<1x4x1440x1440xf32> {
  %cst_2 = arith.constant dense<5.0> : tensor<f32>

  %1 = "tfl.fill"(%arg1, %cst_2) : (tensor<4xi64>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  %36 = tfl.div %1, %arg0 {fused_activation_function = "NONE"} : tensor<1x4x1440x1440xf32>
  return %36 : tensor<1x4x1440x1440xf32>
  // CHECK: %cst = arith.constant dense<5.000000e+00> : tensor<f32>
  // CHECK: %0 = tfl.div(%cst, %arg0) <{fused_activation_function = "NONE"}> : (tensor<f32>, tensor<1x4x1440x1440xf32>) -> tensor<1x4x1440x1440xf32>
  // CHECK: return %0 : tensor<1x4x1440x1440xf32>
}

// CHECK-LABEL: FoldFillOpIntoMulOp
func.func @FoldFillOpIntoMulOp(%arg0: tensor<1x4x1440x1440xf32>, %arg1: tensor<4xi64>) -> tensor<1x4x1440x1440xf32> {
  %cst_2 = arith.constant dense<5.0> : tensor<f32>

  %1 = "tfl.fill"(%arg1, %cst_2) : (tensor<4xi64>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  %36 = tfl.mul %arg0, %1 {fused_activation_function = "NONE"} : tensor<1x4x1440x1440xf32>
  return %36 : tensor<1x4x1440x1440xf32>
  // CHECK: %cst = arith.constant dense<5.000000e+00> : tensor<f32>
  // CHECK: %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<1x4x1440x1440xf32>, tensor<f32>) -> tensor<1x4x1440x1440xf32>
  // CHECK: return %0 : tensor<1x4x1440x1440xf32>
}

// CHECK-LABEL: FuseFillOpIntoFloorModRhs
func.func @FuseFillOpIntoFloorModRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.floor_mod"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.floor_mod"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoFloorModLhs
func.func @FuseFillOpIntoFloorModLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.floor_mod"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.floor_mod"(%cst, %arg0) : (tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoMinimumRhs
func.func @FuseFillOpIntoMinimumRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<5.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.minimum"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<5.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.minimum"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoMinimumLhs
func.func @FuseFillOpIntoMinimumLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.minimum"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.minimum"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoMaximumRhs
func.func @FuseFillOpIntoMaximumRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.maximum"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.maximum"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoMaximumLhs
func.func @FuseFillOpIntoMaximumLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.maximum"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.maximum"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoLessRhs
func.func @FuseFillOpIntoLessRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.less"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.less(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoLessLhs
func.func @FuseFillOpIntoLessLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.less"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.less(%cst, %arg0) : (tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoLessEqualRhs
func.func @FuseFillOpIntoLessEqualRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.less_equal"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.less_equal(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoLessEqualLhs
func.func @FuseFillOpIntoLessEqualLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.less_equal"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.less_equal(%cst, %arg0) : (tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoGreaterRhs
func.func @FuseFillOpIntoGreaterRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.greater"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.greater(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoGreaterLhs
func.func @FuseFillOpIntoGreaterLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.greater"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.greater(%cst, %arg0) : (tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoGreaterEqualRhs
func.func @FuseFillOpIntoGreaterEqualRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.greater_equal"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.greater_equal(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoGreaterEqualLhs
func.func @FuseFillOpIntoGreaterEqualLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.greater_equal"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.greater_equal(%cst, %arg0) : (tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoNotEqualRhs
func.func @FuseFillOpIntoNotEqualRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.not_equal"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.not_equal(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoNotEqualLhs
func.func @FuseFillOpIntoNotEqualLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.not_equal"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.not_equal(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoEqualRhs
func.func @FuseFillOpIntoEqualRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.equal"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.equal"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoEqualLhs
func.func @FuseFillOpIntoEqualLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.equal"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.equal"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoPowRhs
func.func @FuseFillOpIntoPowRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.pow"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.pow(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoPowLhs
func.func @FuseFillOpIntoPowLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.pow"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.pow(%cst, %arg0) : (tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoSquaredDifferenceRhs
func.func @FuseFillOpIntoSquaredDifferenceRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.squared_difference"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.squared_difference(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoSquaredDifferenceLhs
func.func @FuseFillOpIntoSquaredDifferenceLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.squared_difference"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.squared_difference(%cst, %arg0) : (tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoFloorDivRhs
func.func @FuseFillOpIntoFloorDivRhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.floor_div"(%arg0, %0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.floor_div(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoFloorDivLhs
func.func @FuseFillOpIntoFloorDivLhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg1, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.floor_div"(%0, %arg0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.floor_div(%cst, %arg0) : (tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

//===----------------------------------------------------------------------===//
// Fuse broadcast-like ops into select_v2.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: FuseSplatConstantsIntoSelectV2Rhs
func.func @FuseSplatConstantsIntoSelectV2Rhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x1x128x1024xi1>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.select_v2"(%arg1, %arg0, %cst_0) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %arg0, %cst) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoSelectV2Lhs
func.func @FuseSplatConstantsIntoSelectV2Lhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x1x128x1024xi1>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.select_v2"(%arg1, %cst_0, %arg0) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %cst, %arg0) : (tensor<1x1x128x1024xi1>, tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

func.func @FuseSplatConstantsIntoSelectV2LhsInt8(%arg0: tensor<1x8x128x1024xi8>, %arg1: tensor<1x1x128x1024xi1>) -> (tensor<1x8x128x1024xi8>) {
  %cst_0 = arith.constant dense<0> : tensor<1x8x128x1024xi8>
  %1 = "tfl.select_v2"(%arg1, %cst_0, %arg0) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xi8>, tensor<1x8x128x1024xi8>) -> tensor<1x8x128x1024xi8>
  return %1 : tensor<1x8x128x1024xi8>
  // CHECK:  %cst = arith.constant dense<0> : tensor<i8>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %cst, %arg0) : (tensor<1x1x128x1024xi1>, tensor<i8>, tensor<1x8x128x1024xi8>) -> tensor<1x8x128x1024xi8>
  // CHECK:  return %0 : tensor<1x8x128x1024xi8>
}

// CHECK-LABEL: FuseFillOpIntoSelectV2Rhs
func.func @FuseFillOpIntoSelectV2Rhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x1x128x1024xi1>, %arg2: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg2, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.select_v2"(%arg1, %arg0, %0) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %arg0, %cst) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoSelectV2Lhs
func.func @FuseFillOpIntoSelectV2Lhs(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x1x128x1024xi1>, %arg2: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tfl.fill"(%arg2, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  %1 = "tfl.select_v2"(%arg1, %0, %arg0) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %cst, %arg0) : (tensor<1x1x128x1024xi1>, tensor<f32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseFillOpIntoSelectV2LhsInt8
func.func @FuseFillOpIntoSelectV2LhsInt8(%arg0: tensor<1x8x128x1024xi8>, %arg1: tensor<1x1x128x1024xi1>, %arg2: tensor<1x8x128x1024xi8>) -> (tensor<1x8x128x1024xi8>) {
  %cst_0 = arith.constant dense<5> : tensor<i8>
  %0 = "tfl.fill"(%arg2, %cst_0) : (tensor<1x8x128x1024xi8>, tensor<i8>) -> tensor<1x8x128x1024xi8>
  %1 = "tfl.select_v2"(%arg1, %0, %arg0) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xi8>, tensor<1x8x128x1024xi8>) -> tensor<1x8x128x1024xi8>
  return %1 : tensor<1x8x128x1024xi8>
  // CHECK:  %cst = arith.constant dense<5> : tensor<i8>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %cst, %arg0) : (tensor<1x1x128x1024xi1>, tensor<i8>, tensor<1x8x128x1024xi8>) -> tensor<1x8x128x1024xi8>
  // CHECK:  return %0 : tensor<1x8x128x1024xi8>
}

// CHECK-LABEL: FuseBroadcastToIntoSelectV2_WithTwoBroadcastingDims
func.func @FuseBroadcastToIntoSelectV2_WithTwoBroadcastingDims(%arg0: tensor<1x8x128x1024xf32>, %arg1: tensor<1x1x128x1024xi1>) -> (tensor<1x8x128x1024xf32>) {
  %cst = arith.constant dense<[1, 8, 128, 1024]> : tensor<4xi64>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.broadcast_to"(%arg1, %cst) : (tensor<1x1x128x1024xi1>, tensor<4xi64>) -> tensor<1x8x128x1024xi1>
  %2 = "tfl.select"(%1, %arg0, %cst_1) : (tensor<1x8x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %2 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.select_v2"(%arg1, %arg0, %cst) : (tensor<1x1x128x1024xi1>, tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}


// CHECK-LABEL: FuseBroadcastToIntoSelectCondition
func.func @FuseBroadcastToIntoSelectCondition(%arg0: tensor<1x8x1024x2048xf32>, %arg1: tensor<1x8x1024x2048xf32>, %arg2: tensor<1x1x1x2048xi1>) -> (tensor<1x8x1024x2048xf32>, tensor<1x8x1024x2048xf32>) {
  %cst_0 = arith.constant dense<[1, 8, 1024, 2048]> : tensor<4xi32>
  %0 = "tfl.broadcast_to"(%arg2, %cst_0) : (tensor<1x1x1x2048xi1>, tensor<4xi32>) -> tensor<1x8x1024x2048xi1>
  %1 = "tfl.select"(%0, %arg0, %arg1) : (tensor<1x8x1024x2048xi1>, tensor<1x8x1024x2048xf32>, tensor<1x8x1024x2048xf32>) -> tensor<1x8x1024x2048xf32>
  %2 = "tfl.select_v2"(%0, %arg0, %arg1) : (tensor<1x8x1024x2048xi1>, tensor<1x8x1024x2048xf32>, tensor<1x8x1024x2048xf32>) -> tensor<1x8x1024x2048xf32>
  func.return %1, %2 : tensor<1x8x1024x2048xf32>, tensor<1x8x1024x2048xf32>
  // CHECK: %0 = "tfl.select_v2"(%arg2, %arg0, %arg1) : (tensor<1x1x1x2048xi1>, tensor<1x8x1024x2048xf32>, tensor<1x8x1024x2048xf32>) -> tensor<1x8x1024x2048xf32>
  // CHECK: %1 = "tfl.select_v2"(%arg2, %arg0, %arg1) : (tensor<1x1x1x2048xi1>, tensor<1x8x1024x2048xf32>, tensor<1x8x1024x2048xf32>) -> tensor<1x8x1024x2048xf32>
  // CHECK: return %0, %1
}

// CHECK-LABEL: FuseBroadcastToIntoSelectLhs
func.func @FuseBroadcastToIntoSelectLhs(%arg0: tensor<1x10x3xf32>, %arg1: tensor<4x10x3xf32>) -> (tensor<4x10x3xf32>) {
  %cst = arith.constant dense<[4, 10, 3]> : tensor<3xi64>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x10x3xf32>
  %0 = tfl.not_equal(%arg1, %cst_0) : (tensor<4x10x3xf32>, tensor<4x10x3xf32>) -> tensor<4x10x3xi1>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x10x3xf32>, tensor<3xi64>) -> tensor<4x10x3xf32>
  %2 = "tfl.select"(%0, %cst_0, %1) : (tensor<4x10x3xi1>, tensor<4x10x3xf32>, tensor<4x10x3xf32>) -> tensor<4x10x3xf32>
  return %2 : tensor<4x10x3xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.not_equal(%arg1, %cst) : (tensor<4x10x3xf32>, tensor<f32>) -> tensor<4x10x3xi1>
  // CHECK:  %1 = "tfl.select_v2"(%0, %cst, %arg0) : (tensor<4x10x3xi1>, tensor<f32>, tensor<1x10x3xf32>) -> tensor<4x10x3xf32>
  // CHECK:  return %1 : tensor<4x10x3xf32>
}

// CHECK-LABEL: FuseBroadcastToIntoSelectRhs
func.func @FuseBroadcastToIntoSelectRhs(%arg0: tensor<1x10x3xf32>, %arg1: tensor<4x10x3xf32>) -> (tensor<4x10x3xf32>) {
  %cst = arith.constant dense<[4, 10, 3]> : tensor<3xi64>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x10x3xf32>
  %0 = tfl.not_equal(%arg1, %cst_0) : (tensor<4x10x3xf32>, tensor<4x10x3xf32>) -> tensor<4x10x3xi1>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x10x3xf32>, tensor<3xi64>) -> tensor<4x10x3xf32>
  %2 = "tfl.select"(%0, %1, %cst_0) : (tensor<4x10x3xi1>, tensor<4x10x3xf32>, tensor<4x10x3xf32>) -> tensor<4x10x3xf32>
  return %2 : tensor<4x10x3xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.not_equal(%arg1, %cst) : (tensor<4x10x3xf32>, tensor<f32>) -> tensor<4x10x3xi1>
  // CHECK:  %1 = "tfl.select_v2"(%0, %arg0, %cst) : (tensor<4x10x3xi1>, tensor<1x10x3xf32>, tensor<f32>) -> tensor<4x10x3xf32>
  // CHECK:  return %1 : tensor<4x10x3xf32>
}

// CHECK-LABEL: FuseBroadcastToIntoSelect1
func.func @FuseBroadcastToIntoSelect1(%arg0: tensor<1x1x8x1024x2048xf32>, %arg1: tensor<1x1x8x1024x2048xf32>, %arg2: tensor<1x1x1x1x2048xi1>) -> tensor<1x1x8x1024x2048xf32> {
  %cst_0 = arith.constant dense<[1, 1, 8, 1024, 2048]> : tensor<5xi32>
  %0 = "tfl.broadcast_to"(%arg2, %cst_0) : (tensor<1x1x1x1x2048xi1>, tensor<5xi32>) -> tensor<1x1x8x1024x2048xi1>
  %1 = "tfl.select"(%0, %arg0, %arg1) : (tensor<1x1x8x1024x2048xi1>, tensor<1x1x8x1024x2048xf32>, tensor<1x1x8x1024x2048xf32>) -> tensor<1x1x8x1024x2048xf32>

  func.return %1 : tensor<1x1x8x1024x2048xf32>
  // CHECK: %0 = "tfl.select_v2"(%arg2, %arg0, %arg1) : (tensor<1x1x1x1x2048xi1>, tensor<1x1x8x1024x2048xf32>, tensor<1x1x8x1024x2048xf32>) -> tensor<1x1x8x1024x2048xf32>
  // CHECK: return %0
}

//===----------------------------------------------------------------------===//
// Fuse splat constants into binary ops without fused activation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: FuseSplatConstantsIntoMinimumRhs
func.func @FuseSplatConstantsIntoMinimumRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.minimum"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.minimum"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoMaximumRhs
func.func @FuseSplatConstantsIntoMaximumRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.maximum"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.maximum"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoLessRhs
func.func @FuseSplatConstantsIntoLessRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.less"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.less(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoLessEqualRhs
func.func @FuseSplatConstantsIntoLessEqualRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.less_equal"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.less_equal(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoGreaterRhs
func.func @FuseSplatConstantsIntoGreaterRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.greater"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.greater(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoGreaterEqualRhs
func.func @FuseSplatConstantsIntoGreaterEqualRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.greater_equal"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.greater_equal(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoNotEqualRhs
func.func @FuseSplatConstantsIntoNotEqualRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.not_equal"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.not_equal(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoEqualRhs
func.func @FuseSplatConstantsIntoEqualRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.equal"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.equal"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoPowRhs
func.func @FuseSplatConstantsIntoPowRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.pow"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.pow(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoSquaredDifferenceRhs
func.func @FuseSplatConstantsIntoSquaredDifferenceRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.squared_difference"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.squared_difference(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoFloorDivRhs
func.func @FuseSplatConstantsIntoFloorDivRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.floor_div"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = tfl.floor_div(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: FuseSplatConstantsIntoFloorModRhs
func.func @FuseSplatConstantsIntoFloorModRhs(%arg0: tensor<1x8x128x1024xf32>) -> (tensor<1x8x128x1024xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x8x128x1024xf32>
  %1 = "tfl.floor_mod"(%arg0, %cst_0) : (tensor<1x8x128x1024xf32>, tensor<1x8x128x1024xf32>) -> tensor<1x8x128x1024xf32>
  return %1 : tensor<1x8x128x1024xf32>
  // CHECK:  %cst = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.floor_mod"(%arg0, %cst) : (tensor<1x8x128x1024xf32>, tensor<f32>) -> tensor<1x8x128x1024xf32>
  // CHECK:  return %0 : tensor<1x8x128x1024xf32>
}

// CHECK-LABEL: @FuseFullyConnectedAddWithSplat2D
func.func @FuseFullyConnectedAddWithSplat2D(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<40x40xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40x40xf32>) -> tensor<40x40xf32>

  func.return %1 : tensor<40x40xf32>

  // CHECK:  %cst = arith.constant dense<2.000000e+00> : tensor<f32>
  // CHECK:  %0 = "tfl.no_value"() <{value}> : () -> none
  // CHECK:  %1 = "tfl.fully_connected"(%arg0, %arg1, %0) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> tensor<40x40xf32>
  // CHECK:  %2 = tfl.add(%1, %cst) <{fused_activation_function = "NONE"}> : (tensor<40x40xf32>, tensor<f32>) -> tensor<40x40xf32>
  // CHECK:  return %2 : tensor<40x40xf32>
}

// CHECK-LABEL: @fuseMulIntoConv2d_Splat2D
func.func @fuseMulIntoConv2d_Splat2D(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0]]], [[[3.0, 4.0]]]]> : tensor<2x1x1x2xf32>
  %cst1 = arith.constant dense<1.0> : tensor<2xf32>
  %cst2 = arith.constant dense<2.0> : tensor<1x112x112x2xf32>
  %0 = "tfl.conv_2d"(%arg0, %cst0, %cst1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x112x2xf32>, tensor<2x1x1x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x112x112x2xf32>, tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32>

  func.return %1 : tensor<1x112x112x2xf32>
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00]]], {{\[\[\[}}3.000000e+00, 4.000000e+00]]]]> : tensor<2x1x1x2xf32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST3:.*]] = arith.constant dense<1.000000e+00> : tensor<2xf32>
  // CHECK: %[[CONV_RES1:[0-9].*]] = "tfl.conv_2d"(%arg0, %[[CST1]], %[[CST3]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x112x112x2xf32>, tensor<2x1x1x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  // CHECK: %[[RES:[0-9].*]] = tfl.mul(%[[CONV_RES1]], %[[CST2]]) <{fused_activation_function = "NONE"}> : (tensor<1x112x112x2xf32>, tensor<f32>) -> tensor<1x112x112x2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @AvoidFuseFullyConnectedAddWithSplat2D
func.func @AvoidFuseFullyConnectedAddWithSplat2D(%arg0: tensor<1x1x1x1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1x1x1x1xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<1x1x1x1x1xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x1x1x1x1xf32>, tensor<1x1xf32>, none) -> tensor<1x1x1x1x1xf32>
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>) -> tensor<1x1x1x1x1xf32>

  func.return %1 : tensor<1x1x1x1x1xf32>

  // CHECK-DAG: %[[CST1:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<2.000000e+00> : tensor<1x1x1x1x1xf32>
  // CHECK: %[[FC_RESULT:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[CST1]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<1x1x1x1x1xf32>, tensor<1x1xf32>, none) -> tensor<1x1x1x1x1xf32>
  // CHECK: %[[ADD:.*]] = tfl.add %[[FC_RESULT]], %[[CST2]] {fused_activation_function = "NONE"} : tensor<1x1x1x1x1xf32>
  // CHECK: return %[[ADD]] : tensor<1x1x1x1x1xf32>
}

// CHECK-LABEL: @DontFuseMulIntoFullyConnectedForLargeFilter
func.func @DontFuseMulIntoFullyConnectedForLargeFilter(%arg0: tensor<128x256000xf32>) -> tensor<128x1024xf32> {
  %cst0 = arith.constant dense<2.0> : tensor<1024x256000xf32>
  %cst1 = arith.constant dense<2.0> : tensor<1024xf32>
  %cst2 = arith.constant dense<2.0> : tensor<1024xf32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<128x256000xf32>, tensor<1024x256000xf32>, tensor<1024xf32>) -> tensor<128x1024xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<128x1024xf32>, tensor<1024xf32>) -> tensor<128x1024xf32>

  func.return %1 : tensor<128x1024xf32>

// CHECK:  %[[a:.*]] = "tfl.fully_connected"(%arg0, %cst_0, %cst_1) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
// CHECK:  %[[b:.*]] = tfl.mul(%[[a]], %cst) <{fused_activation_function = "RELU6"}>
}
