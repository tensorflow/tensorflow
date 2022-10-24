func.func @test(%V__0: tensor<?x?xf32> { python_test_attrs.static_type = tensor<7x6xf32> }) -> tensor<?x?xf32> {
  %0 = "tf.Asinh"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "tf.FloorDiv"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "tf.Exp"(%1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tf.MulNoNan"(%0, %2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "tf.Xlogy"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "tf.Rsqrt"(%4) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "tf.Softsign"(%5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = "tf.RealDiv"(%3, %6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = "tf.Log1p"(%7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = "tf.FloorDiv"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = "tf.FloorMod"(%9, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = "tf.Reciprocal"(%10) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = "tf.Reciprocal"(%11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = "tf.Maximum"(%8, %12) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = "tf.Asinh"(%13) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %15 = "tf.FloorMod"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = "tf.Xlogy"(%14, %15) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %16 : tensor<?x?xf32>
}
