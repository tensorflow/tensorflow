// RUN: litert-opt -tfl-optimize-broadcast-like='unsafe-fuse-dynamic-shaped-broadcast=false' -split-input-file %s | FileCheck %s
// RUN: litert-opt -tfl-optimize-broadcast-like='unsafe-fuse-dynamic-shaped-broadcast=true' -split-input-file %s | FileCheck --check-prefix=UNSAFE-DYNAMIC-CHECK %s

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
func.func @broadcast_eq(%arg0: tensor<5x7xf32>, %arg1: tensor<7xf32>) -> tensor<5x7xi1> {
  %cst = mhlo.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg1, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tfl.equal"(%arg0, %0) : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xi1>
  func.return %1 : tensor<5x7xi1>
  // CHECK: %0 = "tfl.equal"(%arg0, %arg1) : (tensor<5x7xf32>, tensor<7xf32>) -> tensor<5x7xi1>
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

// CHECK-LABEL: FuseBroadcastToLhsOfDivIntoRhsOfAdd
func.func @FuseBroadcastToLhsOfDivIntoRhsOfAdd(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.div(%1, %arg1) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.add(%arg2, %2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMulIntoRhsOfAdd_quantized
func.func @FuseBroadcastToLhsOfMulIntoRhsOfAdd_quantized(%arg0: tensor<1x1x1x2x1x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>, %arg1: tensor<1x1x1x2x1x!quant.uniform<i8:f32, 0.13491056859493256:61>>, %arg2: tensor<1x1x1x2x1x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>) -> tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.045444928109645844:20>> {
  %cst = arith.constant dense<[1, 1, 1, 2, 64]> : tensor<5xi64>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x1x1x2x1x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>, tensor<5xi64>) -> tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>
  %2 = tfl.mul(%arg1, %1) <{fused_activation_function = "NONE"}> : (tensor<1x1x1x2x1x!quant.uniform<i8:f32, 0.13491056859493256:61>>, tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>) -> tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.045444928109645844:20>>
  %3 = tfl.add(%arg2, %2) {fused_activation_function = "NONE"} : (tensor<1x1x1x2x1x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>, tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.045444928109645844:20>>) -> tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.045444928109645844:20>>
  return %3 : tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.045444928109645844:20>>
  // CHECK:  %cst = arith.constant dense<[1, 1, 1, 2, 64]> : tensor<5xi64>
  // CHECK:  %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x1x1x2x1x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>, tensor<5xi64>) -> tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>
  // CHECK:  %1 = tfl.mul(%arg1, %0) <{fused_activation_function = "NONE"}> : (tensor<1x1x1x2x1x!quant.uniform<i8:f32, 0.13491056859493256:61>>, tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>) -> tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.045444928109645844:20>>
  // CHECK:  %2 = tfl.add(%arg2, %1) <{fused_activation_function = "NONE"}> : (tensor<1x1x1x2x1x!quant.uniform<i8:f32, 0.0033858942333608866:-128>>, tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.045444928109645844:20>>) -> tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.045444928109645844:20>>
  // CHECK:  return %2 : tensor<1x1x1x2x64x!quant.uniform<i8:f32, 0.045444928109645844:20>>
}

// CHECK-LABEL: FuseBroadcastToLhsOfDivIntoRhsOfAdd_neg
func.func @FuseBroadcastToLhsOfDivIntoRhsOfAdd_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.div(%1, %arg1) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.add(%arg2, %2) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}


// CHECK-LABEL: FuseBroadcastToLhsOfDivIntoLhsOfAdd
func.func @FuseBroadcastToLhsOfDivIntoLhsOfAdd(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.div(%1, %arg1) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.add(%2, %arg2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfDivIntoLhsOfAdd_neg
func.func @FuseBroadcastToLhsOfDivIntoLhsOfAdd_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.div(%1, %arg1) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.add(%2, %arg2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfDivIntoRhsOfAdd
func.func @FuseBroadcastToRhsOfDivIntoRhsOfAdd(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.div(%arg1, %1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = tfl.add(%arg2, %2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMulIntoRhsOfAdd_neg
func.func @FuseBroadcastToRhsOfMulIntoRhsOfAdd_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%arg1, %1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = tfl.add(%arg2, %2) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}


// CHECK-LABEL: FuseBroadcastToRhsOfMulIntoLhsOfAdd
func.func @FuseBroadcastToRhsOfMulIntoLhsOfAdd(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%arg1, %1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = tfl.add(%2, %arg2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMulIntoLhsOfAdd_neg
func.func @FuseBroadcastToRhsOfMulIntoLhsOfAdd_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%arg1, %1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = tfl.add(%2, %arg2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMulIntoRhsOfMin
func.func @FuseBroadcastToLhsOfMulIntoRhsOfMin(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%1, %arg1) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%arg2, %2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMulIntoRhsOfMin_neg
func.func @FuseBroadcastToLhsOfMulIntoRhsOfMin_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%1, %arg1) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%arg2, %2) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMulIntoLhsOfMin
func.func @FuseBroadcastToLhsOfMulIntoLhsOfMin(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%1, %arg1) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMulIntoLhsOfMin_neg
func.func @FuseBroadcastToLhsOfMulIntoLhsOfMin_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%1, %arg1) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMulIntoRhsOfMin
func.func @FuseBroadcastToRhsOfMulIntoRhsOfMin(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%arg1, %1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%arg2, %2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMulIntoRhsOfMin_neg
func.func @FuseBroadcastToRhsOfMulIntoRhsOfMin_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%arg1, %1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%arg2, %2) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMulIntoLhsOfMin
func.func @FuseBroadcastToRhsOfMulIntoLhsOfMin(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%arg1, %1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMulIntoLhsOfMin_neg
func.func @FuseBroadcastToRhsOfMulIntoLhsOfMin_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%arg1, %1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMulIntoRhsOfMinWithActFn
func.func @FuseBroadcastToLhsOfMulIntoRhsOfMinWithActFn(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%1, %arg1) {fused_activation_function = "RELU"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%arg2, %2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMulIntoRhsOfMinWithActFn_neg
func.func @FuseBroadcastToLhsOfMulIntoRhsOfMinWithActFn_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%1, %arg1) {fused_activation_function = "RELU"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%arg2, %2) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMulIntoLhsOfMinWithActFn
func.func @FuseBroadcastToLhsOfMulIntoLhsOfMinWithActFn(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%1, %arg1) {fused_activation_function = "RELU"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMulIntoLhsOfMinWithActFn_neg
func.func @FuseBroadcastToLhsOfMulIntoLhsOfMinWithActFn_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = tfl.mul(%1, %arg1) {fused_activation_function = "RELU"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.minimum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoRhsOfMul
func.func @FuseBroadcastToLhsOfMinIntoRhsOfMul(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%arg2, %2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoRhsOfMul_neg
func.func @FuseBroadcastToLhsOfMinIntoRhsOfMul_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%arg2, %2) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoLhsOfMul
func.func @FuseBroadcastToLhsOfMinIntoLhsOfMul(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%2, %arg2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoLhsOfMul_neg
func.func @FuseBroadcastToLhsOfMinIntoLhsOfMul_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%2, %arg2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMinIntoRhsOfMul
func.func @FuseBroadcastToRhsOfMinIntoRhsOfMul(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%arg1, %1) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%arg2, %2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMinIntoRhsOfMul_neg
func.func @FuseBroadcastToRhsOfMinIntoRhsOfMul_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%arg1, %1) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%arg2, %2) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMinIntoLhsOfMul
func.func @FuseBroadcastToRhsOfMinIntoLhsOfMul(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%arg1, %1) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%2, %arg2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMinIntoLhsOfMul_neg
func.func @FuseBroadcastToRhsOfMinIntoLhsOfMul_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%arg1, %1) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%2, %arg2) {fused_activation_function = "NONE"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoRhsOfMulWithActFn
func.func @FuseBroadcastToLhsOfMinIntoRhsOfMulWithActFn(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%arg2, %2) {fused_activation_function = "RELU"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoRhsOfMulWithActFn_neg
func.func @FuseBroadcastToLhsOfMinIntoRhsOfMulWithActFn_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%arg2, %2) {fused_activation_function = "RELU"} : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoLhsOfMulWithActFn
func.func @FuseBroadcastToLhsOfMinIntoLhsOfMulWithActFn(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%2, %arg2) {fused_activation_function = "RELU"} : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoLhsOfMulWithActFn_neg
func.func @FuseBroadcastToLhsOfMinIntoLhsOfMulWithActFn_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = tfl.mul(%2, %arg2) {fused_activation_function = "RELU"} : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoRhsOfMax
func.func @FuseBroadcastToLhsOfMinIntoRhsOfMax(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.maximum"(%arg2, %2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoRhsOfMax_neg
func.func @FuseBroadcastToLhsOfMinIntoRhsOfMax_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.maximum"(%arg2, %2) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoLhsOfMax
func.func @FuseBroadcastToLhsOfMinIntoLhsOfMax(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.maximum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToLhsOfMinIntoLhsOfMax_neg
func.func @FuseBroadcastToLhsOfMinIntoLhsOfMax_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%1, %arg1) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  %3 = "tfl.maximum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMinIntoRhsOfMax
func.func @FuseBroadcastToRhsOfMinIntoRhsOfMax(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%arg1, %1) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = "tfl.maximum"(%arg2, %2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMinIntoRhsOfMax_neg
func.func @FuseBroadcastToRhsOfMinIntoRhsOfMax_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%arg1, %1) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = "tfl.maximum"(%arg2, %2) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMinIntoLhsOfMax
func.func @FuseBroadcastToRhsOfMinIntoLhsOfMax(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<25x32x1xf32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%arg1, %1) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = "tfl.maximum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: FuseBroadcastToRhsOfMinIntoLhsOfMax_neg
func.func @FuseBroadcastToRhsOfMinIntoLhsOfMax_neg(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<25x32x1xf32> {
  %cst = arith.constant dense<[25, 32, 1]> : tensor<3xi32>
  %1 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<f32>, tensor<3xi32>) -> tensor<25x32x1xf32>
  %2 = "tfl.minimum"(%arg1, %1) : (tensor<f32>, tensor<25x32x1xf32>) -> tensor<25x32x1xf32>
  %3 = "tfl.maximum"(%2, %arg2) : (tensor<25x32x1xf32>, tensor<f32>) -> tensor<25x32x1xf32>
  return %3 : tensor<25x32x1xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: @broadcast_add_sub
func.func @broadcast_add_sub(%arg0: tensor<5x7xf32>, %arg1: tensor<7xf32>) -> (tensor<5x7xf32>, tensor<5x7xf32>) {
  %cst = mhlo.constant dense<[5, 7]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg1, %cst) : (tensor<7xf32>, tensor<2xi32>) -> tensor<5x7xf32>
  %1 = "tfl.add"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  %3 = "tfl.sub"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<5x7xf32>
  func.return %1, %3 : tensor<5x7xf32>, tensor<5x7xf32>
  // CHECK-NOT: tfl.broadcast_to
}

// CHECK-LABEL: @broadcast_add_neg
func.func @broadcast_add_neg(%arg0: tensor<2x2xf32>, %arg1: tensor<4x2xf32>, %arg2: tensor<f32>) -> (tensor<2x2xf32>, tensor<4x2xf32>) {
  %cst = mhlo.constant dense<[2, 2]> : tensor<2xi32>
  %cst1 = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.broadcast_to"(%arg2, %cst) : (tensor<f32>, tensor<2xi32>) -> tensor<2x2xf32>
  %1 = "tfl.add"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = "tfl.fully_connected"(%arg1, %0, %cst1) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  func.return %1, %2 : tensor<2x2xf32>, tensor<4x2xf32>
  // CHECK: tfl.broadcast_to
}

// CHECK-LABEL: @broadcast_abs
func.func @broadcast_abs(%arg0: tensor<1x2xf32>) -> (tensor<2x2xf32>) {
  %cst = mhlo.constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  %1 = "tfl.abs"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
  // CHECK: %[[constant:.*]] = mhlo.constant dense<2> : tensor<2xi32>
  // CHECK: %[[abs_value:.*]] = "tfl.abs"(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[broadcasted:.*]] = "tfl.broadcast_to"(%[[abs_value]], %[[constant]]) : (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  // CHECK: return %[[broadcasted]]
}

// CHECK-LABEL: @broadcast_cast
func.func @broadcast_cast(%arg0: tensor<1x2xi8>) -> (tensor<2x2xf32>) {
  %cst = mhlo.constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x2xi8>, tensor<2xi32>) -> tensor<2x2xi8>
  %1 = "tfl.cast"(%0) : (tensor<2x2xi8>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
  // CHECK: %[[constant:.*]] = mhlo.constant dense<2> : tensor<2xi32>
  // CHECK: %[[cast_value:.*]] = "tfl.cast"(%arg0) : (tensor<1x2xi8>) -> tensor<1x2xf32>
  // CHECK: %[[broadcasted:.*]] = "tfl.broadcast_to"(%[[cast_value]], %[[constant]]) : (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  // CHECK: return %[[broadcasted]]
}

// CHECK-LABEL: @broadcast_dequantize
func.func @broadcast_dequantize(%arg0: tensor<1x2x!quant.uniform<i8:f32, 0.0123456789:-128>>) -> (tensor<2x2xf32>) {
  %cst = mhlo.constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x2x!quant.uniform<i8:f32, 0.0123456789:-128>>, tensor<2xi32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.0123456789:-128>>
  %1 = "tfl.dequantize"(%0) : (tensor<2x2x!quant.uniform<i8:f32, 0.0123456789:-128>>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
  // CHECK: %[[constant:.*]] = mhlo.constant dense<2> : tensor<2xi32>
  // CHECK: %[[dequantized:.*]] = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<i8:f32, 0.0123456789:-128>>) -> tensor<1x2xf32>
  // CHECK: %[[broadcasted:.*]] = "tfl.broadcast_to"(%[[dequantized]], %[[constant]]) : (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  // CHECK: return %[[broadcasted]]
}

// CHECK-LABEL: @broadcast_floor
func.func @broadcast_floor(%arg0: tensor<1x2xf32>) -> (tensor<2x2xf32>) {
  %cst = mhlo.constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  %1 = "tfl.floor"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
  // CHECK: %[[constant:.*]] = mhlo.constant dense<2> : tensor<2xi32>
  // CHECK: %[[floor_value:.*]] = "tfl.floor"(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[broadcasted:.*]] = "tfl.broadcast_to"(%[[floor_value]], %[[constant]]) : (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  // CHECK: return %[[broadcasted]]
}

// CHECK-LABEL: @broadcast_zeros_like
func.func @broadcast_zeros_like(%arg0: tensor<1x2xf32>) -> (tensor<2x2xf32>) {
  %cst = mhlo.constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  %1 = "tfl.zeros_like"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
  // CHECK: %[[constant:.*]] = mhlo.constant dense<2> : tensor<2xi32>
  // CHECK: %[[zeros:.*]] = "tfl.zeros_like"(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[broadcasted:.*]] = "tfl.broadcast_to"(%[[zeros]], %[[constant]]) : (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  // CHECK: return %[[broadcasted]]
}

// CHECK-LABEL: @broadcast_mul_dynamic_rhs
func.func @broadcast_mul_dynamic_rhs(%arg0: tensor<?x7xf32>, %arg1: tensor<1x7xf32>) -> tensor<?x7xf32> {
  %shape = "tfl.shape"(%arg0) : (tensor<?x7xf32>) -> tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg1, %shape) : (tensor<1x7xf32>, tensor<2xi32>) -> tensor<?x7xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<?x7xf32>, tensor<?x7xf32>) -> tensor<?x7xf32>
  func.return %1 : tensor<?x7xf32>
  // UNSAFE-DYNAMIC-CHECK: %0 = tfl.mul(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<?x7xf32>, tensor<1x7xf32>) -> tensor<?x7xf32>
}

// CHECK-LABEL: @broadcast_mul_dynamic_rhs2
func.func @broadcast_mul_dynamic_rhs2(%arg0: tensor<?x7xf32>, %arg1: tensor<7xf32>) -> tensor<?x7xf32> {
  %shape = "tfl.shape"(%arg0) : (tensor<?x7xf32>) -> tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg1, %shape) : (tensor<7xf32>, tensor<2xi32>) -> tensor<?x7xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<?x7xf32>, tensor<?x7xf32>) -> tensor<?x7xf32>
  func.return %1 : tensor<?x7xf32>
  // UNSAFE-DYNAMIC-CHECK: %0 = tfl.mul(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<?x7xf32>, tensor<7xf32>) -> tensor<?x7xf32>
}

// CHECK-LABEL: @broadcast_mul_dynamic_lhs
func.func @broadcast_mul_dynamic_lhs(%arg0: tensor<1x7xf32>, %arg1: tensor<?x7xf32>) -> tensor<?x7xf32> {
  %shape = "tfl.shape"(%arg1) : (tensor<?x7xf32>) -> tensor<2xi32>
  %0 = "tfl.broadcast_to"(%arg0, %shape) : (tensor<1x7xf32>, tensor<2xi32>) -> tensor<?x7xf32>
  %1 = "tfl.mul"(%0, %arg1) {fused_activation_function = "NONE"} : (tensor<?x7xf32>, tensor<?x7xf32>) -> tensor<?x7xf32>
  func.return %1 : tensor<?x7xf32>
  // UNSAFE-DYNAMIC-CHECK: %0 = tfl.mul(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x7xf32>, tensor<?x7xf32>) -> tensor<?x7xf32>
}

// CHECK-LABEL: @move_broadcast_through_sum
func.func @move_broadcast_through_sum(%arg0: tensor<1x1x40x100x40x3xf32>) -> tensor<1x4x100x40x3xf32> {
  %cst_0 = arith.constant dense<[1, 4, 40, 100, 40, 3]> : tensor<6xi64>
  %cst_1 = arith.constant dense<2> : tensor<1xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst_0) : (tensor<1x1x40x100x40x3xf32>, tensor<6xi64>) -> tensor<1x4x40x100x40x3xf32>
  %1 = "tfl.sum"(%0, %cst_1) <{keep_dims = false}> : (tensor<1x4x40x100x40x3xf32>, tensor<1xi32>) -> tensor<1x4x100x40x3xf32>
  return %1 : tensor<1x4x100x40x3xf32>
  // CHECK: %cst = arith.constant dense<[1, 4, 100, 40, 3]> : tensor<5xi32>
  // CHECK: %cst_0 = arith.constant dense<2> : tensor<1xi32>
  // CHECK: %0 = "tfl.sum"(%arg0, %cst_0) <{keep_dims = false}> : (tensor<1x1x40x100x40x3xf32>, tensor<1xi32>) -> tensor<1x1x100x40x3xf32>
  // CHECK: %1 = "tfl.broadcast_to"(%0, %cst) : (tensor<1x1x100x40x3xf32>, tensor<5xi32>) -> tensor<1x4x100x40x3xf32>
  // CHECK: return %1 : tensor<1x4x100x40x3xf32>
}

// CHECK-LABEL: @move_broadcast_through_sum_keep_dims
func.func @move_broadcast_through_sum_keep_dims(%arg0: tensor<1x1x40x100x40x3xf32>) -> tensor<1x4x1x100x40x3xf32> {
  %cst_0 = arith.constant dense<[1, 4, 40, 100, 40, 3]> : tensor<6xi64>
  %cst_1 = arith.constant dense<2> : tensor<1xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst_0) : (tensor<1x1x40x100x40x3xf32>, tensor<6xi64>) -> tensor<1x4x40x100x40x3xf32>
  %1 = "tfl.sum"(%0, %cst_1) <{keep_dims = true}> : (tensor<1x4x40x100x40x3xf32>, tensor<1xi32>) -> tensor<1x4x1x100x40x3xf32>
  return %1 : tensor<1x4x1x100x40x3xf32>
  // CHECK: %cst = arith.constant dense<[1, 4, 1, 100, 40, 3]> : tensor<6xi32>
  // CHECK: %cst_0 = arith.constant dense<2> : tensor<1xi32>
  // CHECK: %0 = "tfl.sum"(%arg0, %cst_0) <{keep_dims = true}> : (tensor<1x1x40x100x40x3xf32>, tensor<1xi32>) -> tensor<1x1x1x100x40x3xf32>
  // CHECK: %1 = "tfl.broadcast_to"(%0, %cst) : (tensor<1x1x1x100x40x3xf32>, tensor<6xi32>) -> tensor<1x4x1x100x40x3xf32>
  // CHECK: return %1 : tensor<1x4x1x100x40x3xf32>
}

// CHECK-LABEL: @move_broadcast_through_sum_neg
func.func @move_broadcast_through_sum_neg(%arg0: tensor<1x1x40x100x40x3xf32>) -> tensor<1x40x100x40x3xf32> {
  %cst_0 = arith.constant dense<[1, 4, 40, 100, 40, 3]> : tensor<6xi64>
  %cst_1 = arith.constant dense<1> : tensor<1xi32>
  %0 = "tfl.broadcast_to"(%arg0, %cst_0) : (tensor<1x1x40x100x40x3xf32>, tensor<6xi64>) -> tensor<1x4x40x100x40x3xf32>
  %1 = "tfl.sum"(%0, %cst_1) <{keep_dims = false}> : (tensor<1x4x40x100x40x3xf32>, tensor<1xi32>) -> tensor<1x40x100x40x3xf32>
  return %1 : tensor<1x40x100x40x3xf32>
  // CHECK: %cst = arith.constant dense<[1, 4, 40, 100, 40, 3]> : tensor<6xi64>
  // CHECK: %cst_0 = arith.constant dense<1> : tensor<1xi32>
  // CHECK: %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<1x1x40x100x40x3xf32>, tensor<6xi64>) -> tensor<1x4x40x100x40x3xf32>
  // CHECK: %1 = "tfl.sum"(%0, %cst_0) <{keep_dims = false}> : (tensor<1x4x40x100x40x3xf32>, tensor<1xi32>) -> tensor<1x40x100x40x3xf32>
  // CHECK: return %1 : tensor<1x40x100x40x3xf32>
}
