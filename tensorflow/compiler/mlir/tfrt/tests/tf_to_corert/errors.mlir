// RUN: tf-tfrt-opt -tf-to-tfrt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{failed to legalize operation 'builtin.func' that was explicitly marked illegal}}
func @test_identity_wrong_type(%arg0: tensor<4x2x!tf_type.string>) -> tensor<4x2x!tf_type.stringref> {
  %0 = "tf.SomeOp"(%arg0) : (tensor<4x2x!tf_type.string>) -> tensor<4x2x!tf_type.stringref>
  return %0 : tensor<4x2x!tf_type.stringref>
}
