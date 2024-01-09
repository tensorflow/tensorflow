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
