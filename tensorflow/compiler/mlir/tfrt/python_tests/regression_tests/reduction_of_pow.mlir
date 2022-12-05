func.func @test(%V__0 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<22x50xi32> }) -> tensor<?x?xi32> {
  %0 = "tf.Pow"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims1 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1 = "tf.Sum"(%0, %dims1) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  func.return %1 : tensor<?x?xi32>
}
