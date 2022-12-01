func.func @test(%V__0 : tensor<i1> { python_test_attrs.static_type = tensor<i1> }) -> tensor<i1> {
  %0 = "tf.Cast"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>) -> tensor<i64>
  %1 = "tf.Cast"(%0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>) -> tensor<i1>
  %2 = "tf.Cast"(%1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>) -> tensor<i64>
  %3 = "tf.Less"(%0, %2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %4 = "tf.LogicalOr"(%V__0, %3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
