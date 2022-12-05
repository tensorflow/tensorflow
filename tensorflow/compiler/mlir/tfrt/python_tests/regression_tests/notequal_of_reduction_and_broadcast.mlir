func.func @test(%V__0 : tensor<?x?x?x?xi64> { python_test_attrs.static_type = tensor<26x99x1x1xi64> }) -> tensor<?x?x?x?xi1> {
  %0 = "tf.Const"() { value = dense<[26, 99, 23, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %1 = "tf.BroadcastTo"(%V__0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<4xi32>) -> tensor<?x?x?x?xi64>
  %2 = "tf.AddV2"(%1, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  %dims3 = "tf.Const"() { value = dense<[3]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3 = "tf.Mean"(%2, %dims3) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<1xi32>) -> tensor<?x?x?x?xi64>
  %4 = "tf.NotEqual"(%3, %2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi1>
  func.return %4 : tensor<?x?x?x?xi1>
}
