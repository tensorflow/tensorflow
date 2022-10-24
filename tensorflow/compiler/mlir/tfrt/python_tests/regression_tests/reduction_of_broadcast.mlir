func.func @test(%V__0: tensor<f32> { python_test_attrs.static_type = tensor<f32> }, %V__1: tensor<2xi32> { python_test_attrs.shape_value = dense<[5, 7]> : tensor<2xi32> }) -> tensor<?xf32> {
  %0 = "tf.BroadcastTo"(%V__0, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims1 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1 = "tf.Max"(%0, %dims1) {keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}
