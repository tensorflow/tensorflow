func.func @test(%V__0 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }) -> tensor<f32> {
  %0 = "tf.Square"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
