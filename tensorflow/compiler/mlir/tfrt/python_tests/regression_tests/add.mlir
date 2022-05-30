func.func @test(%V__0: tensor<?x?xf32> { python_test_attrs.static_type = tensor<10x1xf32> }, %V__1: tensor<?xf32> { python_test_attrs.static_type = tensor<5xf32> }) -> tensor<?x?xf32> {
  %2 = "tf.Add"(%V__0, %V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  func.return %2 : tensor<?x?xf32>
}
