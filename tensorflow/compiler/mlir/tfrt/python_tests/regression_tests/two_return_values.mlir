func.func @test(%arg0: tensor<1x?xf32> { python_test_attrs.static_type = tensor<1x128xf32> })
            -> (tensor<1x?xf32>, tensor<1x?xf32>) {
  %c = "tf.Const"() { value = dense<1.000000e+00> : tensor<f32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" }
       : () -> tensor<f32>
  %0 = "tf.Sub"(%c, %arg0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" }
       : (tensor<f32>, tensor<1x?xf32>) -> tensor<1x?xf32>
  %1 = "tf.Sub"(%c, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" }
       : (tensor<f32>, tensor<1x?xf32>) -> tensor<1x?xf32>
  func.return %0, %1 : tensor<1x?xf32>, tensor<1x?xf32>
}
