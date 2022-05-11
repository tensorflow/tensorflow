func.func @test(%V__0: tensor<?x?xf32> { python_test_attrs.static_type = tensor<10x5xf32> }) -> tensor<?x?xf32> {
  %1 = "tf.AddV2"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "tf.FloorMod"(%1, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tf.AddV2"(%2, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %3 : tensor<?x?xf32>
}
