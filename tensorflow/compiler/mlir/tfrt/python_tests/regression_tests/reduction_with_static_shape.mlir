func.func @test(%V__0 : tensor<20x8x11x331xi64> { python_test_attrs.static_type = tensor<20x8x11x331xi64> }) -> tensor<1x8x11x331xi64> {
  %dims3 = "tf.Const"() { value = dense<[0]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3 = "tf.Prod"(%V__0, %dims3) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<20x8x11x331xi64>, tensor<1xi32>) -> tensor<1x8x11x331xi64>
  func.return %3 : tensor<1x8x11x331xi64>
}
