func.func @test(%V__0 : tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1xf32> }, %V__1 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }) -> tensor<?x?x?xf32> {
  %0 = "tf.Const"() { value = dense<[1, 1, 1]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1 = "tf.Reshape"(%V__1, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %2 = "tf.Polygamma"(%V__0, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "tf.Erfc"(%2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %4 = "tf.Squeeze"(%V__1) { squeeze_dims = [ 0 : i64, 2 : i64, 3 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?xf32>
  %5 = "tf.BiasAdd"(%3, %4) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  func.return %5 : tensor<?x?x?xf32>
}
