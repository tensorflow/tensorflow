func.func @test(%V__0 : tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }, %V__1 : tensor<2xi32> { python_test_attrs.shape_value = dense<[42, 71]> : tensor<2xi32> }) -> tensor<?x?xf32> {
  %0 = "tf.Acosh"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Zeta"(%0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Polygamma"(%V__0, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.BroadcastTo"(%2, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %3 : tensor<?x?xf32>
}
