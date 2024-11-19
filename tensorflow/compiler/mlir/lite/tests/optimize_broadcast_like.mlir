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
