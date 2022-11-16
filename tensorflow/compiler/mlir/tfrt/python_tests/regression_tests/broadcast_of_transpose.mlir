func.func @test(%V__0 : tensor<?x?xi64> { python_test_attrs.static_type = tensor<1x36xi64> }) -> tensor<?x?xi64> {
  %dims0 = "tf.Const"() { value = dense<[1, 0]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %0 = "tf.Transpose"(%V__0, %dims0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %1 = "tf.Const"() { value = dense<[36, 5]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2 = "tf.BroadcastTo"(%0, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  func.return %2 : tensor<?x?xi64>
}
