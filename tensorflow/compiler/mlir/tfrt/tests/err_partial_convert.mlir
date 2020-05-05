// RUN: tf-opt %s -tf-legalize-to-hex  -verify-diagnostics

func @partial_convert() {
  %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{failed to legalize operation 'tf.Const'}}
  %1 = "tf.Const"() {value = dense<42> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return
}
