func.func @test(%V__0: tensor<?x?xf32> { python_test_attrs.static_type = tensor<10x5xf32> }) -> tensor<?x?xf32> {
  %1 = "tf.Acos"(%V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}
