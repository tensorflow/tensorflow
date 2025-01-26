// RUN: mlir-hlo-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// CHECK-LABEL: constant_like_constant
func.func @constant_like_constant(%arg0: tensor<3x4xi32>) -> tensor<3x4xf32> {
  // CHECK: chlo.constant dense<3.200000e+00>
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<3x4xi32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: constant_like_constant_dynamic
func.func @constant_like_constant_dynamic(%arg0: tensor<?x?xi32>) -> tensor<?x?xf32> {
  // CHECK: chlo.constant_like
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<?x?xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
