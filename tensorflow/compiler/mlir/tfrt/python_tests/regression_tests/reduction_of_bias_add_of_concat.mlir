func.func @test(%V__0 : tensor<?x?xf32> { python_test_attrs.static_type = tensor<96x48xf32> }, %V__1 : tensor<?xf32> { python_test_attrs.static_type = tensor<96xf32> }) -> tensor<?xf32> {
  %axis0 = "tf.Const"() { value = dense<1> : tensor<i32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<i32>
  %0 = "tf.ConcatV2"(%V__0, %V__0, %axis0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<i32>) -> tensor<?x?xf32>
  %1 = "tf.BiasAdd"(%0, %V__1) { data_format = "NHWC", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  %dims1 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2 = "tf.Prod"(%1, %dims1) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %2 : tensor<?xf32>
}
