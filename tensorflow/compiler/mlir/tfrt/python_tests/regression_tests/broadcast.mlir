func.func @test(%V__0: tensor<?xf32> { python_test_attrs.static_type = tensor<10xf32> }, %V__1: tensor<2xi32> { python_test_attrs.shape_value = dense<[5,10]> : tensor<2xi32> }) -> tensor<?x?xf32> {
  %0 = "tf.BroadcastTo"(%V__0, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
