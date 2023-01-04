func.func @test(%V__0 : tensor<?xi1> { python_test_attrs.static_type = tensor<1xi1> }, %V__1 : tensor<?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x1xi64> }) -> tensor<?x?x?xi64> {
  %dims0 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %0 = "tf.Min"(%V__1, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?xi64>
  %1 = "tf.Select"(%V__0, %V__1, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
  %dims2 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2 = "tf.Mean"(%1, %dims2) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?xi64>
  func.return %2 : tensor<?x?x?xi64>
}
