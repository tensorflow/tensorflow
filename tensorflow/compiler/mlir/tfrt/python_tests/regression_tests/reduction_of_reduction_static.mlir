func.func @test(%V__0 : tensor<85x65x31x40xf32> { python_test_attrs.static_type = tensor<85x65x31x40xf32> }) -> tensor<1xf32> {
  %dims0 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Sum"(%V__0, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<85x65x31x40xf32>, tensor<1xi32>) -> tensor<1x65x31x40xf32>
  %dims1 = "tf.Const"() { value = dense<[1, 2, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %1 = "tf.Max"(%0, %dims1) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1x65x31x40xf32>, tensor<3xi32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}
