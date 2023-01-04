func.func @test(%V__0 : tensor<66x76x122xui16> { python_test_attrs.static_type = tensor<66x76x122xui16> }) -> tensor<66x76xui16> {
  %0 = "tf.OnesLike"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<66x76x122xui16>) -> tensor<66x76x122xui16>
  %dims1 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1 = "tf.Sum"(%0, %dims1) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<66x76x122xui16>, tensor<1xi32>) -> tensor<66x76xui16>
  func.return %1 : tensor<66x76xui16>
}
