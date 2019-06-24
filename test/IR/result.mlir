// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test mixed normal and variadic results
//===----------------------------------------------------------------------===//

func @correct_variadic_result() -> tensor<f32> {
  // CHECK: mixed_normal_variadic_result
  %0:5 = "test.mixed_normal_variadic_result"() : () -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
  return %0#4 : tensor<f32>
}

// -----

func @error_in_first_variadic_result() -> tensor<f32> {
  // expected-error @+1 {{result #1 must be tensor of any type}}
  %0:5 = "test.mixed_normal_variadic_result"() : () -> (tensor<f32>, f32, tensor<f32>, tensor<f32>, tensor<f32>)
  return %0#4 : tensor<f32>
}

// -----

func @error_in_normal_result() -> tensor<f32> {
  // expected-error @+1 {{result #2 must be tensor of any type}}
  %0:5 = "test.mixed_normal_variadic_result"() : () -> (tensor<f32>, tensor<f32>, f32, tensor<f32>, tensor<f32>)
  return %0#4 : tensor<f32>
}

// -----

func @error_in_second_variadic_result() -> tensor<f32> {
  // expected-error @+1 {{result #3 must be tensor of any type}}
  %0:5 = "test.mixed_normal_variadic_result"() : () -> (tensor<f32>, tensor<f32>, tensor<f32>, f32, tensor<f32>)
  return %0#4 : tensor<f32>
}

