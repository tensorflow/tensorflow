builtin.func @test(%V__0 : tensor<i32> { python_test_attrs.static_type = tensor<i32> }) -> tensor<i32> {
  %0 = "tf.OnesLike"(%V__0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.ZerosLike"(%V__0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Div"(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %2 : tensor<i32>
}
