builtin.func @test(%V__0 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<1x1xi32> }, %V__1 : tensor<0xi32> { python_test_attrs.shape_value = dense<[]> : tensor<0xi32> }) -> tensor<i32> {
  %0 = "tf.Reshape"(%V__0, %V__1) : (tensor<?x?xi32>, tensor<0xi32>) -> tensor<i32>
  return %0 : tensor<i32>
}
