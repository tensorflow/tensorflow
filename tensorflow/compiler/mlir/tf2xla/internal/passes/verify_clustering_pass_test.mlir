// RUN: tf-opt -verify-clustering-pass  -split-input-file -verify-diagnostics %s | FileCheck %s
// Tests the VerifyClusteringPass Pass, ensures that an error is thrown when validation fails.

func.func @testNotTfDialect(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
 // expected-error@below {{op is in dialect chlo not in tf functional dialect}}
  %0 = "chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = array<i64: 3>} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}

// -----

// CHECK-LABEL: func @testTFDialect
func.func @testTFDialect(%arg0: tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string> {
  %0 = "tf.Identity"(%arg0) : (tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string>
  func.return %0 : tensor<4x2x!tf_type.string>
}


// -----

func.func @testTFDialect(%arg0: tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string> {
   // expected-error@below {{op has outside compilation attribute _xla_outside_compilation which is not allowed after clustering}}
  %0 = "tf.Identity"(%arg0) {_xla_outside_compilation = "cluster1"}: (tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string>
  func.return %0 : tensor<4x2x!tf_type.string>
}

