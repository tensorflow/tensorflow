// RUN: tf-opt %s -split-input-file -xla-legalize-to-std | FileCheck %s

// -----

// CHECK-LABEL: func @reshape.const.1() -> tensor<f32> {
func @reshape.const.1() -> tensor<f32> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<f32>
  %cst = constant  {name = "constant.1"} dense<42.0> : tensor<1x1xf32>
  %0 = "xla.reshape"(%cst) : (tensor<1x1xf32>) -> tensor<f32>
  // CHECK-NEXT: return %cst : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @reshape.const.2() -> tensor<2xf32> {
func @reshape.const.2() -> tensor<2xf32> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<2xf32>
  %cst = constant  {name = "constant.1"} dense<42.0> : tensor<1x2xf32>
  %0 = "xla.reshape"(%cst) : (tensor<1x2xf32>) -> tensor<2xf32>
  // CHECK-NEXT: return %cst : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @reshape.const.3() -> tensor<1xf32> {
func @reshape.const.3() -> tensor<1xf32> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<1xf32>
  %cst = constant  {name = "constant.1"} dense<42.0> : tensor<f32>
  %0 = "xla.reshape"(%cst) : (tensor<f32>) -> tensor<1xf32>
  // CHECK-NEXT: return %cst : tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @reshape.const.4() -> tensor<16xi64> {
func @reshape.const.4() -> tensor<16xi64> {
  // CHECK-NEXT: %cst = constant dense<42> : tensor<16xi64>
  %cst = constant  dense<42> : tensor<4x4xi64>
  %0 = "xla.reshape"(%cst) : (tensor<4x4xi64>) -> tensor<16xi64>
  // CHECK-NEXT: return %cst : tensor<16xi64>
  return %0 : tensor<16xi64>
}

// -----

// CHECK-LABEL: func @reshape.const.5() -> tensor<16xf64> {
func @reshape.const.5() -> tensor<16xf64> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<16xf64>
  %cst = constant  dense<4.200000e+01> : tensor<4x4xf64>
  %0 = "xla.reshape"(%cst) : (tensor<4x4xf64>) -> tensor<16xf64>
  // CHECK-NEXT: return %cst : tensor<16xf64>
  return %0 : tensor<16xf64>
}


// -----

// CHECK-LABEL: func @reshape.const.6() -> tensor<6xi32> {
func @reshape.const.6() -> tensor<6xi32> {
  // CHECK-NEXT: %cst = constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %cst = constant  {name = "constant.1"} dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %0 = "xla.reshape"(%cst) : (tensor<3x2xi32>) -> tensor<6xi32>
  // CHECK-NEXT: return %cst : tensor<6xi32>
  return %0 : tensor<6xi32>
}


// -----

// CHECK-LABEL: func @reshape.const.7() -> tensor<2x3xi32> {
func @reshape.const.7() -> tensor<2x3xi32> {
  // CHECK-NEXT: %cst = constant dense<{{\[\[}}1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %cst = constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %0 = "xla.reshape"(%cst) : (tensor<6xi32>) -> tensor<2x3xi32>
  // CHECK-NEXT: return %cst : tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}