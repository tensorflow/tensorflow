func.func @test(%V__0 : tensor<?x?xf32> { python_test_attrs.static_type = tensor<1x59xf32> }) -> tensor<?x?xf32> {
  %dims0 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Max"(%V__0, %dims0) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  %1 = "tf.Softmax"(%0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Const"() { value = dense<[23, 1]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %3 = "tf.BroadcastTo"(%1, %2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %3 : tensor<?x?xf32>
}
