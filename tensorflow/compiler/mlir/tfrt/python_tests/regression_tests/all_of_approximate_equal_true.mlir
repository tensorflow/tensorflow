func.func @test(%V__0: tensor<?x?xf32>
    { python_test_attrs.static_type = tensor<48x4xf32> }) -> tensor<?xi1> {
  %0 = "tf.ApproximateEqual"(%V__0, %V__0) {
        tolerance = 1.000000e-05 : f32,
        device = "/job:localhost/replica:0/task:0/device:CPU:0"
      } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  %dims1 = "tf.Const"() {
        value = dense<[1]> : tensor<1xi32>,
        device = "/job:localhost/replica:0/task:0/device:CPU:0"
      } : () -> tensor<1xi32>
  %1 = "tf.All"(%0, %dims1) {
        keep_dims = false,
        device = "/job:localhost/replica:0/task:0/device:CPU:0"
      } : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  func.return %1 : tensor<?xi1>
}
