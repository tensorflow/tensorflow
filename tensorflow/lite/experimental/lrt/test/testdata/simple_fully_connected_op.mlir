module {
func.func @main(%arg0: tensor<1x128x2304xf32>, %arg1: tensor<2048x2304xf32>) -> tensor<1x128x2048xf32> {
  %cst = "tfl.no_value"() <{value}> : () -> none
  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) <{fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT", asymmetric_quantize_inputs = false}> : (tensor<1x128x2304xf32>, tensor<2048x2304xf32>, none) -> tensor<1x128x2048xf32>
  return %0 : tensor<1x128x2048xf32>
}
}