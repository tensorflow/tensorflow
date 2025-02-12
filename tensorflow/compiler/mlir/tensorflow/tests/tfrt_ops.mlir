// RUN: tf-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Tests for TensorFlow TFRT ops with custom verifiers.

//===--------------------------------------------------------------------===//
//  Test TF operations (tf.*)
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func @testPwStreamResults
func.func @testPwStreamResults(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  "tf.PwStreamResults"(%arg0, %arg1) {names = ["foo", "bar"]} : (tensor<f32>, tensor<f32>) -> ()
  return
}

// -----
// CHECK-LABEL: func @test_ifrt_call
func.func @test_ifrt_call(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  %result = "tf.IfrtCall"(%arg0, %arg1) <{program_id = 1234 : i64, variable_arg_indices = [0 : i32, 1 : i32], variable_names = ["a", "b"]}> : (tensor<?xf32>, tensor<?xf32>) -> (tensor<1x1xf32>)
  func.return
}

// -----
func.func @test_ifrt_call_fail_unsorted_variable_arg_indices(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // expected-error@below {{variable_arg_indices must be sorted in ascending order}}
  %result = "tf.IfrtCall"(%arg0, %arg1) <{program_id = 1234 : i64, variable_arg_indices = [1 : i32, 0 : i32], variable_names = ["a", "b"]}> : (tensor<?xf32>, tensor<?xf32>) -> (tensor<1x1xf32>)
  func.return
}
