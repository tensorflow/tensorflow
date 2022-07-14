func.func @test(%V__0 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }) -> tensor<?xf32> {
  %0 = "tf.Const"() { value = dense<[2, 32769]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1 = "tf.Fill"(%0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<2xi32>, tensor<f32>) -> tensor<?x?xf32>
  %dims2 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %2 = "tf.Max"(%1, %dims2) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %2 : tensor<?xf32>
}
