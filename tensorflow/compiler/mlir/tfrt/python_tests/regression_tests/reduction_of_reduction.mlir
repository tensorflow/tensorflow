builtin.func @test(%V__0: tensor<?x?x?xi64>
    { python_test_attrs.static_type = tensor<9x32x26xi64> })
    -> tensor<?x?xi64> {
  %dims1 = "tf.Const"() { value = dense<[0]> : tensor<1xi32> }:
      () -> tensor<1xi32>
  %1 = "tf.Mean"(%V__0, %dims1) {keep_dims = false} :
      (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %dims2 = "tf.Const"() { value = dense<[1]> : tensor<1xi32> }:
      () -> tensor<1xi32>
  %2 = "tf.Min"(%1, %dims2) {keep_dims = true} :
      (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  return %2 : tensor<?x?xi64>
}
