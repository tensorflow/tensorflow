func.func @test(%V__0: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }, %V__1: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }, %V__2: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }, %V__3: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }) -> tensor<?xf32> {
  %0 = "tf.Rint"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.RealDiv"(%V__0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Relu"(%1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Erf"(%2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Erf"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Reciprocal"(%V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Xlogy"(%V__1, %5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Rint"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Abs"(%7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Relu6"(%8) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Xlog1py"(%6, %9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Erf"(%V__3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.ClipByValue"(%4, %10, %11) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.Round"(%12) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  func.return %13 : tensor<?xf32>
}
