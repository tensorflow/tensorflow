// RUN: tf-tfrt-opt -tf-to-tfrt %s -split-input-file -verify-diagnostics

func @test_identity_wrong_type(%arg0: tensor<4x2x!tf_type.string>) -> tensor<4x2x!tf_type.stringref> {
  // expected-warning @+2 {{failed to find a non-empty 'device' attribute}}
  // expected-error @+1 {{failed to legalize operation 'tf.SomeOp' that was explicitly marked illegal}}
  %0 = "tf.SomeOp"(%arg0) : (tensor<4x2x!tf_type.string>) -> tensor<4x2x!tf_type.stringref>
  return %0 : tensor<4x2x!tf_type.stringref>
}
