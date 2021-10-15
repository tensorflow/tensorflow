builtin.func @test(%arg0: tensor<?x?x1xf32>, %arg1: tensor<f32>) -> tensor<?x?x1xi1> {
  %ne = "tf.NotEqual"(%arg0, %arg1) { incompatible_shape_error = false} : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xi1>
  return %ne : tensor<?x?x1xi1>
}
