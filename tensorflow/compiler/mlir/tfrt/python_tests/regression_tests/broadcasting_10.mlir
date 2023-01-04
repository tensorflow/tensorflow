func.func @test(%V__0: tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<10x1x5xf32> }, %V__1: tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<1x4x1xf32> }) -> tensor<?x?x?xf32> {
  %0 = "tf.Atan"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "tf.Erf"(%0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "tf.Sub"(%1, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "tf.Softsign"(%2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %4 = "tf.Reciprocal"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = "tf.Acosh"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %6 = "tf.Floor"(%5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %7 = "tf.Asin"(%6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %8 = "tf.FloorMod"(%7, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %9 = "tf.Sinh"(%8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %10 = "tf.Cos"(%9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %11 = "tf.Atanh"(%10) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %12 = "tf.Atan2"(%V__0, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %13 = "tf.Sin"(%12) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %14 = "tf.Mod"(%11, %13) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %15 = "tf.Atanh"(%14) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %16 = "tf.Sin"(%15) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %17 = "tf.Div"(%4, %16) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %17 : tensor<?x?x?xf32>
}
