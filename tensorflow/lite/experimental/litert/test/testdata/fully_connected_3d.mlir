module {
func.func @main(%arg0: tensor<8x100x128xf32>, %arg1: tensor<128x128xf32>, %arg2: none) -> tensor<8x100x128xf32> {
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<8x100x128xf32>, tensor<128x128xf32>, none) -> tensor<8x100x128xf32>
  return %0 : tensor<8x100x128xf32>
}
}
