func.func @test(%V__0 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }, %V__1 : tensor<1x?x1x1xf32> { python_test_attrs.static_type = tensor<1x1x1x1xf32> }) -> tensor<?x?x?xf32> {
  %0 = "tf.Shape"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1x?x1x1xf32>) -> tensor<4xi32>
  %1 = "tf.Reshape"(%V__0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  %dims2 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2 = "tf.Mean"(%1, %dims2) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  func.return %2 : tensor<?x?x?xf32>
}
