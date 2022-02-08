builtin.func @test(%V__0 : tensor<16x45x?xi32> { python_test_attrs.static_type = tensor<16x45x1xi32> }) -> tensor<?xi32> {
  %dims0 = "tf.Const"() { value = dense<[1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %0 = "tf.Max"(%V__0, %dims0) { keep_dims = true } : (tensor<16x45x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  %1 = "tf.Squeeze"(%0) { squeeze_dims = [ 1 : i64, 2 : i64 ] } : (tensor<?x?x?xi32>) -> tensor<?xi32>
  return %1 : tensor<?xi32>
}
