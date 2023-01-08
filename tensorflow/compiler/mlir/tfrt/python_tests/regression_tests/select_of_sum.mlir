func.func @test(%V__0 : tensor<i1> { python_test_attrs.static_type = tensor<i1> }, %V__1 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<1x81xi32> }, %V__2 : tensor<?x?xi32> { python_test_attrs.static_type = tensor<1x1xi32> }) -> tensor<?x?xi32> {
  %0 = "tf.Sub"(%V__1, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %dims1 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %1 = "tf.Sum"(%0, %dims1) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi32>, tensor<1xi32>) -> tensor<?x?xi32>
  %2 = "tf.Select"(%V__0, %1, %V__2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i1>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  func.return %2 : tensor<?x?xi32>
}
