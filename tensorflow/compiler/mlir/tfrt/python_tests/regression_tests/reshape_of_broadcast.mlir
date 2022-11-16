func.func @test(%V__0 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<60x1x96x26xi64> }, %V__1 : tensor<4xi32> { python_test_attrs.shape_value = dense<[1, 1, 1, 1]> : tensor<4xi32> }) -> tensor<i64> {
  %dims0 = "tf.Const"() { value = dense<[0, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %0 = "tf.Max"(%V__0, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %1 = "tf.BroadcastTo"(%0, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2 = "tf.Const"() { value = dense<[]> : tensor<0xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<0xi32>
  %3 = "tf.Reshape"(%1, %2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<0xi32>) -> tensor<i64>
  func.return %3 : tensor<i64>
}
