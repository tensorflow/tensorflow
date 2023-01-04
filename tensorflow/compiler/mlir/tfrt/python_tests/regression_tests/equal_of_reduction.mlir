func.func @test(%V__0 : tensor<?x?xi64> { python_test_attrs.static_type = tensor<82x8xi64> }) -> tensor<?xi1> {
  %dims0 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Max"(%V__0, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %dims1 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1 = "tf.Sum"(%0, %dims1) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %2 = "tf.Equal"(%1, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1>
  func.return %2 : tensor<?xi1>
}
