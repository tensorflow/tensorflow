func.func @test(%V__0 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<37x18xi32> }, %V__1 : tensor<1x1xi64> { python_test_attrs.static_type = tensor<1x1xi64> }) -> tensor<?x?xi64> {
  %dims0 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %0 = "tf.Sum"(%V__0, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %1 = "tf.Equal"(%0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi1>
  %2 = "tf.SelectV2"(%1, %V__1, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<?x?xi64>
  func.return %2 : tensor<?x?xi64>
}
