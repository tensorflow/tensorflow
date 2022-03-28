func.func @test(%V__0: tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<10x1x1xf32> }) -> tensor<?x?x?xf32> {
  %dims0 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %0 = "tf.Min"(%V__0, %dims0) {keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %1 = "tf.ClipByValue"(%V__0, %0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %1 : tensor<?x?x?xf32>
}
