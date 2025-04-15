// RUN: odml-to-stablehlo-opt %s -unfold-splat-constant-pass -cse -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @unfold_splat_constant_float
func.func @unfold_splat_constant_float() -> tensor<1x750xf32> {
  %cst = mhlo.constant dense<7.680000e+02> : tensor<1x750xf32>
  func.return %cst : tensor<1x750xf32>

  // CHECK-DAG: %0 = mhlo.constant dense<7.680000e+02> : tensor<f32>
  // CHECK: %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<1x750xf32>
  // CHECK: return %1 : tensor<1x750xf32>
}

// CHECK-LABEL: @unfold_splat_constant_integer
func.func @unfold_splat_constant_integer() -> tensor<1x750xi32> {
  %cst = mhlo.constant dense<1> : tensor<1x750xi32>
  func.return %cst : tensor<1x750xi32>

  // CHECK-DAG: %0 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i32>) -> tensor<1x750xi32>
  // CHECK: return %1 : tensor<1x750xi32>
}

// CHECK-LABEL: @splat_scalar_no_change
func.func @splat_scalar_no_change() -> (tensor<f32>, tensor<i32>) {
  // CHECK-NOT: mhlo.broadcast_in_dim
  %cst0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %cst1 = mhlo.constant dense<0> : tensor<i32>
  func.return %cst0, %cst1 : tensor<f32>, tensor<i32>
}
