builtin.func @test(%V__0: tensor<?x?xi64>
    { python_test_attrs.static_type = tensor<48x4xi64> }) -> tensor<?xi1> {
  %0 = "tf.ApproximateEqual"(%V__0, %V__0) {tolerance = 1.000000e-05 : f32} :
      (tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi1>
  %dims1 = "tf.Const"() { value = dense<[1]> : tensor<1xi32> }:
      () -> tensor<1xi32>
  %1 = "tf.All"(%0, %dims1) {keep_dims = false} :
      (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  return %1 : tensor<?xi1>
}
