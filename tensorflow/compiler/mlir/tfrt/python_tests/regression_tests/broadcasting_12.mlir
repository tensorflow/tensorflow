func.func @test(%V__0: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }, %V__1: tensor<2xi32> { python_test_attrs.shape_value = dense<[4, 3]> : tensor<2xi32> }) -> tensor<?x?xf32> {
  %0 = "tf.Asinh"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.BroadcastTo"(%0, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %2 = "tf.Acosh"(%1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tf.Sqrt"(%2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "tf.Asin"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "tf.Round"(%4) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "tf.Sinh"(%5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = "tf.Acosh"(%6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = "tf.Relu6"(%7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = "tf.Ceil"(%8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = "tf.Sin"(%9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %10 : tensor<?x?xf32>
}
