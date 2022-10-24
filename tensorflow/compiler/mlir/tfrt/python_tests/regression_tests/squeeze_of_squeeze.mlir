func.func @test(%V__0 : tensor<?x?x?xi1> { python_test_attrs.static_type = tensor<1x1x1xi1> }) -> tensor<?xi1> {
  %dims0 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.All"(%V__0, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>, tensor<1xi32>) -> tensor<?x?x?xi1>
  %1 = "tf.Squeeze"(%0) { squeeze_dims = [ 2 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi1>) -> tensor<?x?xi1>
  %2 = "tf.Squeeze"(%1) { squeeze_dims = [ 1 : i64 ], device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi1>) -> tensor<?xi1>
  func.return %2 : tensor<?xi1>
}
