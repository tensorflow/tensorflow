module {
func.func @main(%arg0: tensor<1x4x256x128xf32>, %arg1: tensor<1x4x128x128xf32>) -> tensor<1x4x256x128xf32> {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x4x256x128xf32>, tensor<1x4x128x128xf32>) -> tensor<1x4x256x128xf32>
  return %0 : tensor<1x4x256x128xf32>
}
}