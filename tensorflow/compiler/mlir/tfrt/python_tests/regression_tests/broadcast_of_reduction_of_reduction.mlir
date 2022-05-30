func.func @test(%V__0: tensor<?x?x?xi32> { python_test_attrs.static_type = tensor<81x38x24xi32> }, %V__1: tensor<3xi32> { python_test_attrs.shape_value = dense<[67, 7, 73]> : tensor<3xi32> }) -> tensor<?x?x?xi32> {
  %dims1 = "tf.Const"() { value = dense<[0, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1 = "tf.Max"(%V__0, %dims1) {keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  %dims2 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %2 = "tf.Prod"(%1, %dims2) {keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?xi32>
  %3 = "tf.BroadcastTo"(%2, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %3 : tensor<?x?x?xi32>
}
