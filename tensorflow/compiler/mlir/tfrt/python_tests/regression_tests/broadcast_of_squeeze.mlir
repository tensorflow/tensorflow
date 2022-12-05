func.func @test(%V__0 : tensor<?x1x?xi64> { python_test_attrs.static_type = tensor<1x1x1xi64> }, %V__1 : tensor<?x?x1xi1> { python_test_attrs.static_type = tensor<1x60x1xi1> }) -> tensor<?x?x?xi64> {
  %0 = "tf.Squeeze"(%V__0) { squeeze_dims = [ 0 : i64, 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x1x?xi64>) -> tensor<1xi64>
  %1 = "tf.Shape"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x1xi1>) -> tensor<3xi32>
  %2 = "tf.BroadcastTo"(%0, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  func.return %2 : tensor<?x?x?xi64>
}
