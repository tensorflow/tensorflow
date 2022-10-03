func.func @test(%V__0 : tensor<i16> { python_test_attrs.static_type = tensor<i16> }, %V__1 : tensor<i16> { python_test_attrs.static_type = tensor<i16> }) -> tensor<i16> {
  %0 = "tf.Sub"(%V__0, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i16>, tensor<i16>) -> tensor<i16>
  %1 = "tf.Square"(%0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i16>) -> tensor<i16>
  func.return %1 : tensor<i16>
}
