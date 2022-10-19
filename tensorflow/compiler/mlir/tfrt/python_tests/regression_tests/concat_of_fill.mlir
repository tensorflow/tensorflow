func.func @test(%V__0 : tensor<?x?xi1> { python_test_attrs.static_type = tensor<1x2xi1> }, %V__1 : tensor<i1> { python_test_attrs.static_type = tensor<i1> }) -> tensor<?x?xi1> {
  %0 = "tf.Const"() { value = dense<[1, 0]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1 = "tf.Fill"(%0, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<2xi32>, tensor<i1>) -> tensor<?x?xi1>
  %axis2 = "tf.Const"() { value = dense<1> : tensor<i32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<i32>
  %2 = "tf.ConcatV2"(%V__0, %1, %axis2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>, tensor<?x?xi1>, tensor<i32>) -> tensor<?x?xi1>
  func.return %2 : tensor<?x?xi1>
}
