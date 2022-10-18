func.func @test(%V__0 : tensor<?x?xi64> { python_test_attrs.static_type = tensor<1x1xi64> }, %V__1 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1xi64> }, %V__2 : tensor<1x?x1xi32> { python_test_attrs.static_type = tensor<1x32x1xi32> }) -> tensor<?xi64> {
  %3 = "tf.Shape"(%V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1x?x1xi32>) -> tensor<3xi32>
  %4 = "tf.BroadcastTo"(%V__1, %3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %dims5 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %5 = "tf.Sum"(%4, %dims5) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  %dims6 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %6 = "tf.Sum"(%5, %dims6) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %7 = "tf.Shape"(%6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>) -> tensor<2xi32>
  %8 = "tf.Reshape"(%V__0, %7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %9 = "tf.OnesLike"(%8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>) -> tensor<?x?xi64>
  %dims10 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %10 = "tf.Sum"(%9, %dims10) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  func.return %10 : tensor<?xi64>
}
