func.func @test(%V__0: tensor<?x?xf32> { python_test_attrs.static_type = tensor<7x4xf32> }) -> tensor<?x?xf32> {
  %0 = "tf.Sqrt"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "tf.Mul"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "tf.FloorDiv"(%1, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tf.Pow"(%0, %2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "tf.Cos"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "tf.Round"(%4) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "tf.Rsqrt"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = "tf.Abs"(%6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = "tf.FloorDiv"(%7, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = "tf.Div"(%5, %8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = "tf.Asinh"(%9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = "tf.Atanh"(%10) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = "tf.LeakyRelu"(%11) {alpha = 0.3 : f32, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = "tf.Selu"(%12) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = "tf.Ceil"(%13) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %15 = "tf.Maximum"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = "tf.Elu"(%15) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %17 = "tf.Floor"(%16) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %18 = "tf.Xlogy"(%14, %17) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %18 : tensor<?x?xf32>
}
