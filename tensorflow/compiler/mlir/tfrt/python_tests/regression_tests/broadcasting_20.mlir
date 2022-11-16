func.func @test(%V__0: tensor<?x?xf32> { python_test_attrs.static_type = tensor<3x3xf32> }, %V__1: tensor<?xf32> { python_test_attrs.static_type = tensor<3xf32> }, %V__2: tensor<2xi32> { python_test_attrs.shape_value = dense<[3, 3]> : tensor<2xi32> }, %V__3: tensor<?x?xf32> { python_test_attrs.static_type = tensor<1x3xf32> }) -> tensor<?x?xf32> {
  %0 = "tf.Mul"(%V__1, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Sinh"(%0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.FloorMod"(%1, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.BroadcastTo"(%2, %V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %4 = "tf.Cosh"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "tf.Sigmoid"(%4) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "tf.Cos"(%5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = "tf.Minimum"(%V__0, %6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = "tf.Rint"(%7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = "tf.Acosh"(%8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = "tf.Maximum"(%V__0, %9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = "tf.LeakyRelu"(%V__3) {alpha = 0.1 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = "tf.Asin"(%11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = "tf.Floor"(%V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = "tf.Xlogy"(%12, %13) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %15 = "tf.FloorMod"(%10, %14) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %15 : tensor<?x?xf32>
}
