func.func @test(%V__0 : tensor<1x?xf32> { python_test_attrs.static_type = tensor<1x1xf32> }) -> tensor<?x?x?x?xf32> {
  %dims0 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Max"(%V__0, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1x?xf32>, tensor<1xi32>) -> tensor<1x1xf32>
  %1 = "tf.Const"() { value = dense<[81, 98, 1, 18]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %2 = "tf.BroadcastTo"(%0, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1x1xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %3 = "tf.Acosh"(%2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims4 = "tf.Const"() { value = dense<[0, 1, 2]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %4 = "tf.Max"(%3, %dims4) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?x?xf32>
  func.return %4 : tensor<?x?x?x?xf32>
}
