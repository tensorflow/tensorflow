builtin.func @test(%V__0: tensor<?x?xi64> { python_test_attrs.static_type = tensor<1x2xi64> }, %V__1: tensor<?x?xi32> { python_test_attrs.static_type = tensor<1x2xi32> }) -> tensor<?x?xi32> {
  %0 = "tf.Equal"(%V__0, %V__0) {} : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
  %dims3 = "tf.Const"() { value = dense<[0]> : tensor<1xi32> }: () -> tensor<1xi32>
  %3 = "tf.Mean"(%V__1, %dims3) {keep_dims = true} : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %4 = "tf.Select"(%0, %V__1, %3) {} : (tensor<?x?xi1>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  return %4 : tensor<?x?xi32>
}
