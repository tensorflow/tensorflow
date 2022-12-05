func.func @test(%V__0 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<1x1x62x1xi64> }) -> tensor<?x?x?x?xi1> {
  %dims0 = "tf.Const"() { value = dense<[0, 1, 3]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %0 = "tf.Mean"(%V__0, %dims0) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<3xi32>) -> tensor<?x?x?x?xi64>
  %1 = "tf.ClipByValue"(%0, %V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %2 = "tf.BitwiseXor"(%1, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %3 = "tf.NotEqual"(%2, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi1>
  func.return %3 : tensor<?x?x?x?xi1>
}
