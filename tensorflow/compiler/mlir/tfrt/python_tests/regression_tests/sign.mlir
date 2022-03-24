func.func @test(%V__0: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }) -> tensor<?xf32> {
  %0 = "tf.ZerosLike"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.IsNan"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xi1>
  %2 = "tf.Cast"(%1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>) -> tensor<?xf32>
  %3 = "tf.Div"(%0, %2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Sign"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  func.return %4 : tensor<?xf32>
}
