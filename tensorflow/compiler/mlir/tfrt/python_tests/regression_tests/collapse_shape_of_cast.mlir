func.func @test(%V__0 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1xi64> }, %V__1 : tensor<?x1x1x1xi32> { python_test_attrs.static_type = tensor<1x1x1x1xi32> }) -> tensor<?xi64> {
  %0 = "tf.Shape"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x1x1x1xi32>) -> tensor<4xi32>
  %1 = "tf.BroadcastTo"(%V__0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2 = "tf.Max"(%1, %dims2) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims3 = "tf.Const"() { value = dense<[1, 2, 3, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %3 = "tf.Transpose"(%2, %dims3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims4 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %4 = "tf.Sum"(%3, %dims4) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?xi64>
  func.return %4 : tensor<?xi64>
}
