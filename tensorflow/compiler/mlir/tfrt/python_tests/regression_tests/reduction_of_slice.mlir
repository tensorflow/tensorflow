func.func @test(%V__0 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<119x78x68x146xi64> }) -> tensor<?x?x?x?xi64> {
  %begin1 = "tf.Const"() { value = dense<[7, 0, 0, 43]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %size1 = "tf.Const"() { value = dense<[87, 78, 68, 12]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1 = "tf.Slice"(%V__0, %begin1, %size1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %dims2 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2 = "tf.Min"(%1, %dims2) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  func.return %2 : tensor<?x?x?x?xi64>
}
