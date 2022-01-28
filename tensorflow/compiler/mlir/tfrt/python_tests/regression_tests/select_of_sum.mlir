builtin.func @test(%V__0 : tensor<i1> { python_test_attrs.static_type = tensor<i1> }, %V__1 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<1x81xi32> }, %V__2 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<1x1xi32> }) -> tensor<?x?xi32> {
  %0 = "tf.Sub"(%V__1, %V__1) : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims1 = "tf.Const"() { value = dense<[1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.Sum"(%0, %dims1) { keep_dims = true } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %2 = "tf.Select"(%V__0, %1, %V__2) : (tensor<i1>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  return %2 : tensor<?x?xi32>
}
