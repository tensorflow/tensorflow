func.func @test(%V__0 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<1x86xi32> }, %V__1 : tensor<86xi32> { python_test_attrs.static_type = tensor<86xi32> }) -> tensor<?x?xi32> {
  %0 = "tf.ZerosLike"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<86xi32>) -> tensor<86xi32>
  %1 = "tf.Mul"(%V__0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<86xi32>) -> tensor<?x?xi32>
  %dims2 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2 = "tf.Sum"(%1, %dims2) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  func.return %2 : tensor<?x?xi32>
}
