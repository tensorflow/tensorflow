func.func @test(%V__0 : tensor<?x?xf32> { python_test_attrs.static_type = tensor<1x22xf32> }, %V__1 : tensor<?xi1> { python_test_attrs.static_type = tensor<1xi1> }) -> tensor<?x?xf32> {
  %dims0 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Mean"(%V__0, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  %1 = "tf.Select"(%V__1, %V__0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "tf.Sub"(%V__0, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %dims3 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3 = "tf.Mean"(%2, %dims3) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  func.return %3 : tensor<?x?xf32>
}
