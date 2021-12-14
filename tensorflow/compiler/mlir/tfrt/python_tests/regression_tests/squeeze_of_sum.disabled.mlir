builtin.func @test(%V__0 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<75x80x1xi64> }) -> tensor<?xi64> {
  %dims0 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32> } : () -> tensor<2xi32>
  %0 = "tf.Sum"(%V__0, %dims0) { keep_dims = true } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %1 = "tf.Squeeze"(%0) { squeeze_dims = [ 0 : i64, 2 : i64 ] } : (tensor<?x?x?xi64>) -> tensor<?xi64>
  return %1 : tensor<?xi64>
}
