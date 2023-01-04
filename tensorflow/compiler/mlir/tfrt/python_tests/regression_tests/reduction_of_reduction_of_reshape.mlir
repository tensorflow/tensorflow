func.func @test(%V__0 : tensor<i64> { python_test_attrs.static_type = tensor<i64> }, %V__1 : tensor<1x?x1xi32> { python_test_attrs.static_type = tensor<1x1x1xi32> }) -> tensor<?x?x?xi64> {
  %0 = "tf.Shape"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1x?x1xi32>) -> tensor<3xi32>
  %1 = "tf.Reshape"(%V__0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<3xi32>) -> tensor<?x?x?xi64>
  %dims2 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2 = "tf.Sum"(%1, %dims2) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %dims3 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3 = "tf.Mean"(%2, %dims3) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  func.return %3 : tensor<?x?x?xi64>
}
