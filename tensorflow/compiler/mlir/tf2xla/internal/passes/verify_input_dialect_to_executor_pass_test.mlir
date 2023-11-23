// RUN: tf-opt -verify-input-dialect-to-executor-pass  -split-input-file -verify-diagnostics %s | FileCheck %s
// Tests the VerifyClusteringPass Pass, ensures that an error is thrown when validation fails.

// -----

// CHECK-LABEL: func @testNoClusterFuncOpPasses
func.func @testNoClusterFuncOpPasses(%arg0: tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string> {
  %0 = "tf.Identity"(%arg0) : (tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string>
  func.return %0 : tensor<4x2x!tf_type.string>
}

// -----

func.func @testClusterFuncOpFails(%arg0: tensor<i32>) -> tensor<i32> {
   // expected-error@below {{failed TF functional to executor validation, op tf_device.cluster_func is not allowed}}
  %cluster = "tf_device.cluster_func"(%arg0) {func = @_func} : (tensor<i32>) -> tensor<i32>
 func.return %cluster : tensor<i32>
}