builtin.func @test(%V__0 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<34x84x49xi64> }) -> tensor<?x?xi64> {
  %0 = "tf.OnesLike"(%V__0) : (tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims1 = "tf.Const"() { value = dense<[2]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.Prod"(%0, %dims1) { keep_dims = false } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %2 = "tf.OnesLike"(%1) : (tensor<?x?xi64>) -> tensor<?x?xi64>
  %3 = "tf.FloorDiv"(%1, %2) : (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
  return %3 : tensor<?x?xi64>
}
