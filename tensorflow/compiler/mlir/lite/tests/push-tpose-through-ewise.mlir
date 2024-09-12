// RUN: tf-opt %s --push-transpose-through-ewise --split-input-file | FileCheck %s

// CHECK-LABEL: pushTposeAfterAddSimple
func.func @pushTposeAfterAddSimple(%arg0: tensor<2x3x4x5xf32>) -> tensor<5x2x3x4xf32> {
  %perm = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %cst = arith.constant dense<1.0> : tensor<5x2x3x4xf32>
  %1 = tfl.add %0, %cst { fused_activation_function = "NONE" } : tensor<5x2x3x4xf32>
  func.return %1 : tensor<5x2x3x4xf32>
}

// CHECK: %cst = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %cst_0 = arith.constant dense<1.000000e+00> : tensor<2x3x4x5xf32>
// CHECK: %0 = tfl.add %arg0, %cst_0 {fused_activation_function = "NONE"} : tensor<2x3x4x5xf32>
// CHECK: %1 = "tfl.transpose"(%0, %cst) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
// CHECK: return %1 : tensor<5x2x3x4xf32>

// -----

// CHECK-LABEL: pushTposeAfterAddSimpleWithFold
func.func @pushTposeAfterAddSimpleWithFold(%arg0: tensor<2x3xi32>) -> tensor<3x2xi32> {
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  %cst = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %1 = tfl.add %0, %cst { fused_activation_function = "NONE" } : tensor<3x2xi32>
  func.return %1 : tensor<3x2xi32>
}

// CHECK: %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK: [1, 3, 5], [2, 4, 6]
// CHECK: %0 = tfl.add %arg0, %cst_0 {fused_activation_function = "NONE"} : tensor<2x3xi32>
// CHECK: %1 = "tfl.transpose"(%0, %cst) : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
// CHECK: return %1 : tensor<3x2xi32>

// -----

// CHECK-LABEL: pushTposeAfterSubSimple
func.func @pushTposeAfterSubSimple(%arg0: tensor<2x3x4x5xf32>) -> tensor<5x2x3x4xf32> {
  %perm = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %cst = arith.constant dense<1.0> : tensor<5x2x3x4xf32>
  %1 = tfl.sub %0, %cst { fused_activation_function = "NONE" } : tensor<5x2x3x4xf32>
  func.return %1 : tensor<5x2x3x4xf32>
}

// CHECK: %cst = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %cst_0 = arith.constant dense<1.000000e+00> : tensor<2x3x4x5xf32>
// CHECK: %0 = tfl.sub %arg0, %cst_0 {fused_activation_function = "NONE"} : tensor<2x3x4x5xf32>
// CHECK: %1 = "tfl.transpose"(%0, %cst) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
// CHECK: return %1 : tensor<5x2x3x4xf32>

// -----

// CHECK-LABEL: permNotConstNoChange
func.func @permNotConstNoChange(%arg0: tensor<2x3x4x5xf32>, %perm: tensor<4xi32>) -> tensor<5x2x3x4xf32> {
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %cst = arith.constant dense<1.0> : tensor<5x2x3x4xf32>
  %1 = tfl.add %0, %cst { fused_activation_function = "NONE" } : tensor<5x2x3x4xf32>
  func.return %1 : tensor<5x2x3x4xf32>
}

// CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<5x2x3x4xf32>
// CHECK: %0 = "tfl.transpose"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
// CHECK: %1 = tfl.add %0, %cst {fused_activation_function = "NONE"} : tensor<5x2x3x4xf32>
// CHECK: return %1 : tensor<5x2x3x4xf32>

// -----

// CHECK-LABEL: doubleTposeInput
func.func @doubleTposeInput(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x4x5xf32>) -> tensor<5x2x3x4xf32> {
  %perm = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %perm1 = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %1 = "tfl.transpose"(%arg1, %perm1) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %2 = tfl.add %0, %1 { fused_activation_function = "NONE" } : tensor<5x2x3x4xf32>
  func.return %2 : tensor<5x2x3x4xf32>
}

// CHECK: %cst = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x3x4x5xf32>
// CHECK: %1 = "tfl.transpose"(%0, %cst) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
// CHECK: return %1 : tensor<5x2x3x4xf32>

// -----

// CHECK-LABEL: pushTposeBcastNoChange
func.func @pushTposeBcastNoChange(%arg0: tensor<2x3x4x1xf32>) -> tensor<5x2x3x4xf32> {
  %perm = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x3x4x1xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
  %cst = arith.constant dense<1.0> : tensor<5x2x3x4xf32>
  %1 = "tfl.add"(%0, %cst) { fused_activation_function = "NONE" } : (tensor<1x2x3x4xf32>, tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32>
  func.return %1 : tensor<5x2x3x4xf32>
}

// CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<5x2x3x4xf32>
// CHECK: %cst_0 = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %0 = "tfl.transpose"(%arg0, %cst_0) : (tensor<2x3x4x1xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
// CHECK: %1 = tfl.add(%0, %cst) <{fused_activation_function = "NONE"}> : (tensor<1x2x3x4xf32>, tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32>

// -----

// CHECK-LABEL: doubleTposeOneBroadcastInput
func.func @doubleTposeOneBroadcastInput(%arg0: tensor<2x3x4x1xf32>, %arg1: tensor<2x3x4x5xf32>) -> tensor<5x2x3x4xf32> {
  %perm = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x3x4x1xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
  %perm1 = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %1 = "tfl.transpose"(%arg1, %perm1) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %2 = "tfl.add"(%0, %1) { fused_activation_function = "NONE" } : (tensor<1x2x3x4xf32>, tensor<5x2x3x4xf32>) -> tensor<5x2x3x4xf32>
  func.return %2 : tensor<5x2x3x4xf32>
}

// CHECK: %cst = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %0 = tfl.add(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<2x3x4x1xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
// CHECK: %1 = "tfl.transpose"(%0, %cst) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
// CHECK: return %1 : tensor<5x2x3x4xf32>

// -----

// CHECK-LABEL: doubleTposeInputPermNotEqualNoChange
func.func @doubleTposeInputPermNotEqualNoChange(%arg0: tensor<2x4x3x5xf32>, %arg1: tensor<2x3x4x5xf32>) -> tensor<5x2x3x4xf32> {
  %perm = arith.constant dense<[3, 0, 2, 1]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x4x3x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %perm1 = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %1 = "tfl.transpose"(%arg1, %perm1) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %2 = tfl.add %0, %1 { fused_activation_function = "NONE" } : tensor<5x2x3x4xf32>
  func.return %2 : tensor<5x2x3x4xf32>
}

// CHECK: %cst = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %cst_0 = arith.constant dense<[3, 0, 2, 1]> : tensor<4xi32>
// CHECK: %0 = "tfl.transpose"(%arg0, %cst_0) : (tensor<2x4x3x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
// CHECK: %1 = "tfl.transpose"(%arg1, %cst) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
// CHECK: %2 = tfl.add %0, %1 {fused_activation_function = "NONE"} : tensor<5x2x3x4xf32>

// -----

// CHECK-LABEL: pushTposeBcastCstInput
func.func @pushTposeBcastCstInput(%arg0: tensor<2x3x4x5xf32>) -> tensor<5x2x3x4xf32> {
  %perm = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %cst = arith.constant dense<1.0> : tensor<1x2x3x4xf32>
  %1 = "tfl.add"(%0, %cst) { fused_activation_function = "NONE" } : (tensor<5x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<5x2x3x4xf32>
  func.return %1 : tensor<5x2x3x4xf32>
}

// CHECK: %cst = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %cst_0 = arith.constant dense<1.000000e+00> : tensor<2x3x4x1xf32>
// CHECK: %0 = tfl.add(%arg0, %cst_0) <{fused_activation_function = "NONE"}> : (tensor<2x3x4x5xf32>, tensor<2x3x4x1xf32>) -> tensor<2x3x4x5xf32>
// CHECK: %1 = "tfl.transpose"(%0, %cst) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>

// -----

// CHECK-LABEL: pushTposeBcastScalarCstInput
func.func @pushTposeBcastScalarCstInput(%arg0: tensor<2x3x4x5xf32>) -> tensor<5x2x3x4xf32> {
  %perm = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>
  %cst = arith.constant dense<1.0> : tensor<f32>
  %1 = "tfl.add"(%0, %cst) { fused_activation_function = "NONE" } : (tensor<5x2x3x4xf32>, tensor<f32>) -> tensor<5x2x3x4xf32>
  func.return %1 : tensor<5x2x3x4xf32>
}

// CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %cst_0 = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x3x4x5xf32>, tensor<f32>) -> tensor<2x3x4x5xf32>
// CHECK: %1 = "tfl.transpose"(%0, %cst_0) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<5x2x3x4xf32>

// -----

// CHECK-LABEL: pushTposeDynamicBcastScalarCstInput
func.func @pushTposeDynamicBcastScalarCstInput(%arg0: tensor<?x?x4x5xf32>) -> tensor<5x?x?x4xf32> {
  %perm = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<?x?x4x5xf32>, tensor<4xi32>) -> tensor<5x?x?x4xf32>
  %cst = arith.constant dense<1.0> : tensor<f32>
  %1 = "tfl.add"(%0, %cst) { fused_activation_function = "NONE" } : (tensor<5x?x?x4xf32>, tensor<f32>) -> tensor<5x?x?x4xf32>
  func.return %1 : tensor<5x?x?x4xf32>
}

// CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK: %cst_0 = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<?x?x4x5xf32>, tensor<f32>) -> tensor<?x?x4x5xf32>
// CHECK: %1 = "tfl.transpose"(%0, %cst_0) : (tensor<?x?x4x5xf32>, tensor<4xi32>) -> tensor<5x?x?x4xf32>

// -----

// CHECK-LABEL: doubleTposeDynamicInput
func.func @doubleTposeDynamicInput(%arg0: tensor<?x?x4x5xf32>, %arg1: tensor<?x?x4x5xf32>) -> tensor<5x?x?x4xf32> {
  %perm = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %perm) : (tensor<?x?x4x5xf32>, tensor<4xi32>) -> tensor<5x?x?x4xf32>
  %perm1 = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %1 = "tfl.transpose"(%arg1, %perm1) : (tensor<?x?x4x5xf32>, tensor<4xi32>) -> tensor<5x?x?x4xf32>
  %2 = tfl.add %0, %1 { fused_activation_function = "NONE" } : tensor<5x?x?x4xf32>
  func.return %2 : tensor<5x?x?x4xf32>
}

// CHECK: %cst = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK: %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<?x?x4x5xf32>
// CHECK: %1 = "tfl.transpose"(%0, %cst) : (tensor<?x?x4x5xf32>, tensor<4xi32>) -> tensor<5x?x?x4xf32>
// CHECK: return %1 : tensor<5x?x?x4xf32>
