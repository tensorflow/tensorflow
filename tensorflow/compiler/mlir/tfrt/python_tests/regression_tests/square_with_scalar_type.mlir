builtin.func @test(%V__0 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }) -> tensor<f32> {
  %0 = "tf.Square"(%V__0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
