func.func @test(%V__0 : tensor<i32> { python_test_attrs.static_type = tensor<i32> }) -> tensor<i32> {
  %0 = "tf.OnesLike"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %1 = "tf.ZerosLike"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Div"(%0, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}
