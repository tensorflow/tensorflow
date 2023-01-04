func.func @test(%V__0 : tensor<?x1x1xi64> { python_test_attrs.static_type = tensor<99x1x1xi64> }, %V__1 : tensor<?x?xi64> { python_test_attrs.static_type = tensor<99x1xi64> }) -> tensor<?x1xi64> {
  %dims0 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Prod"(%V__0, %dims0) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x1x1xi64>, tensor<1xi32>) -> tensor<?x1xi64>
  %1 = "tf.ClipByValue"(%0, %0, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x1xi64>, tensor<?x1xi64>, tensor<?x?xi64>) -> tensor<?x1xi64>
  func.return %1 : tensor<?x1xi64>
}
