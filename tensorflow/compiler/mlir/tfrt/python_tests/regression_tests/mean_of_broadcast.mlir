func.func @test(%V__0 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<1x1xi32> }) -> tensor<?x?x?xi32> {
  %0 = "tf.Const"() { value = dense<[97, 16, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1 = "tf.BroadcastTo"(%V__0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %2 = "tf.BitwiseOr"(%1, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<?x?xi32>) -> tensor<?x?x?xi32>
  %dims3 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3 = "tf.Mean"(%2, %dims3) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<1xi32>) -> tensor<?x?x?xi32>
  func.return %3 : tensor<?x?x?xi32>
}
