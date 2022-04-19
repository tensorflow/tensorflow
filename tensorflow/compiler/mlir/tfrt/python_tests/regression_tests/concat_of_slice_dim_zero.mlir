func.func @test(%V__0 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<2x2xi32> }) -> tensor<?x?xi32> {
  %begin0 = "tf.Const"() { value = dense<[0, 0]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %size0 = "tf.Const"() { value = dense<[2, 0]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %0 = "tf.Slice"(%V__0, %begin0, %size0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  %axis1 = "tf.Const"() { value = dense<1> : tensor<i32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<i32>
  %1 = "tf.ConcatV2"(%V__0, %0, %axis1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  func.return %1 : tensor<?x?xi32>
}
