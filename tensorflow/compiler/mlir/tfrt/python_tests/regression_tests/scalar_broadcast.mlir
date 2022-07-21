func.func @test(%V__0 : tensor<i1> { python_test_attrs.static_type = tensor<i1> }, %V__1 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }, %V__2 : tensor<f32> { python_test_attrs.static_type = tensor<f32> }) -> tensor<f32> {
  %0 = "tf.Cast"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Selu"(%V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %2 = "tf.NextAfter"(%1, %V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %3 = "tf.Elu"(%2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %4 = "tf.Cosh"(%3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %5 = "tf.Elu"(%4) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>) -> tensor<f32>
  %6 = "tf.Div"(%V__1, %5) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %7 = "tf.Select"(%0, %6, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %7 : tensor<f32>
}
