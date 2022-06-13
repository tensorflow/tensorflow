func.func @test(%V__0 : tensor<?x?x?x?xui64> { python_test_attrs.static_type = tensor<1x22x36x98xui64> }, %V__2 : tensor<?x?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x36x39xi1> }) -> tensor<?x?x?x?xi1> {
  %begin1 = "tf.Const"() { value = dense<[0, 0, 0, 12]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %size1 = "tf.Const"() { value = dense<[1, 1, 36, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1 = "tf.Slice"(%V__0, %begin1, %size1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xui64>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x?x?x?xui64>
  %3 = "tf.NotEqual"(%1, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xui64>, tensor<?x?x?x?xui64>) -> tensor<?x?x?x?xi1>
  %4 = "tf.ZerosLike"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims5 = "tf.Const"() { value = dense<[0, 3]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %5 = "tf.Any"(%4, %dims5) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<2xi32>) -> tensor<?x?x?x?xi1>
  func.return %5 : tensor<?x?x?x?xi1>
}
