builtin.func @test(%V__0 : tensor<?x?x?xi1> { python_test_attrs.static_type = tensor<1x71x1xi1> }, %V__1 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1xi64> }, %V__2 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<94x46x42x7xi64> }) -> tensor<?x?x?xi64> {
  %dims0 = "tf.Const"() { value = dense<[0, 1, 2, 3]> : tensor<4xi32> } : () -> tensor<4xi32>
  %0 = "tf.Min"(%V__2, %dims0) { keep_dims = true } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %1 = "tf.Invert"(%0) : (tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %2 = "tf.Squeeze"(%1) { squeeze_dims = [ 1 : i64 ] } : (tensor<?x?x?x?xi64>) -> tensor<?x?x?xi64>
  %3 = "tf.Select"(%V__0, %V__1, %2) : (tensor<?x?x?xi1>, tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  return %3 : tensor<?x?x?xi64>
}
