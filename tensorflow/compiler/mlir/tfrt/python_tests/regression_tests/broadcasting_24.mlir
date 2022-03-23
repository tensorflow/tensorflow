func.func @test(%V__0 : tensor<?x?x?x?xi1> { python_test_attrs.static_type = tensor<62x1x77x13xi1> }, %V__1 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<62x1x77x13xf32> }) -> tensor<?x?x?xf32> {
  %0 = "tf.OnesLike"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
  %dims1 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1 = "tf.Min"(%V__1, %dims1) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %2 = "tf.Select"(%0, %1, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %3 = "tf.Rint"(%2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %4 = "tf.Round"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims5 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %5 = "tf.Min"(%4, %dims5) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %dims6 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %6 = "tf.Sum"(%5, %dims6) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?x?xf32>
  %dims7 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %7 = "tf.Mean"(%6, %dims7) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?x?xf32>
  %8 = "tf.Atanh"(%7) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims9 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %9 = "tf.Mean"(%8, %dims9) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %10 = "tf.Sign"(%9) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %10 : tensor<?x?x?xf32>
}
