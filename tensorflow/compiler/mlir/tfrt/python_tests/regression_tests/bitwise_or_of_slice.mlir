func.func @test(%V__0 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<0x21xi32> }) -> tensor<?x?xi32> {
  %begin0 = "tf.Const"() { value = dense<[0, 0]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %size0 = "tf.Const"() { value = dense<[0, 21]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %0 = "tf.Slice"(%V__0, %begin0, %size0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %1 = "tf.BitwiseOr"(%0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  func.return %1 : tensor<?x?xi32>
}
