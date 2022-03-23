func.func @test(%V__0: tensor<?xf32> { python_test_attrs.static_type = tensor<10xf32> }) -> tensor<?xf32> {
  %0 = "tf.Sub"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Sqrt"(%0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Pow"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Floor"(%2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Sinh"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Softsign"(%4) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Asin"(%5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Floor"(%6) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Rint"(%7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Atanh"(%8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Erf"(%9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Asin"(%10) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.Floor"(%11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.FloorDiv"(%1, %12) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %13 : tensor<?xf32>
}
