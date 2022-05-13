func.func @test(%V__0 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<48x34x1x76xi64> }) -> tensor<?xi64> {
  %dims0 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Max"(%V__0, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %dims1 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1 = "tf.Max"(%0, %dims1) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<2xi32>) -> tensor<?x?x?x?xi64>
  %dims2 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %2 = "tf.Max"(%1, %dims2) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?xi64>
  func.return %2 : tensor<?xi64>
}
