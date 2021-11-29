builtin.func @test(%V__0: tensor<?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x10xi1> }, %V__1: tensor<?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x10xi32> }, %V__2: tensor<?x?x?xi32> { python_test_attrs.static_type = tensor<1x1x10xi32> }) -> tensor<?x?x?xi32> {
  %dims2 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32> }: () -> tensor<2xi32>
  %0 = "tf.Max"(%V__1, %dims2) {keep_dims = true} : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %1 = "tf.Select"(%V__0, %0, %V__2) {} : (tensor<?x?x?xi1>, tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return %1 : tensor<?x?x?xi32>
}
