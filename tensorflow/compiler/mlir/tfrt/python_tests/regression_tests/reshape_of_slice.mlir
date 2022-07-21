func.func @test(%V__0 : tensor<?xf32> { python_test_attrs.static_type = tensor<95xf32> }, %V__1 : tensor<?x?xf32> { python_test_attrs.static_type = tensor<1x11xf32> }) -> tensor<?x?xf32> {
  %begin0 = "tf.Const"() { value = dense<[26]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %size0 = "tf.Const"() { value = dense<[11]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Slice"(%V__0, %begin0, %size0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  %1 = "tf.Shape"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<2xi32>
  %2 = "tf.Reshape"(%0, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %2 : tensor<?x?xf32>
}
