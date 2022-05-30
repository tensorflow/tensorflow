func.func @test(%arg0: tensor<?x?x1xf32> { python_test_attrs.static_type = tensor<10x5x1xf32> }, %arg1: tensor<f32> { python_test_attrs.static_type = tensor<f32> }) -> tensor<?x?x1xi1> {
  %ne = "tf.NotEqual"(%arg0, %arg1) { incompatible_shape_error = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xi1>
  func.return %ne : tensor<?x?x1xi1>
}
