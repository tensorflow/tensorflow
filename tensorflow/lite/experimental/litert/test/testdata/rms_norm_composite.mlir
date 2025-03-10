module {

func.func @main(%arg0: tensor<1x128x2304xf32>, %arg1: tensor<2304xf32>) -> tensor<1x128x2304xf32> {
  %0 = stablehlo.composite "odml.rms_norm" %arg0, %arg1 {composite_attributes = {epsilon = 9.99999997E-7 : f32}, decomposition = @odml.rms_norm.impl} : (tensor<1x128x2304xf32>, tensor<2304xf32>) -> tensor<1x128x2304xf32>
  return %0 : tensor<1x128x2304xf32>
}

func.func @odml.rms_norm.impl(%arg0: tensor<1x128x2304xf32>, %arg1: tensor<2304xf32>) -> tensor<1x128x2304xf32> {
    %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<1x128x2304xf32>
    %1 = "tfl.pseudo_const"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "tfl.sum"(%0, %1) <{keep_dims = false}> : (tensor<1x128x2304xf32>, tensor<1xi32>) -> tensor<1x128xf32>
    %3 = "tfl.pseudo_const"() <{value = dense<4.34027781E-4> : tensor<f32>}> : () -> tensor<f32>
    %4 = tfl.mul(%2, %3) <{fused_activation_function = "NONE"}> : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x128xf32>
    %5 = "tfl.pseudo_const"() <{value = dense<9.99999997E-7> : tensor<f32>}> : () -> tensor<f32>
    %6 = tfl.add(%4, %5) <{fused_activation_function = "NONE"}> : (tensor<1x128xf32>, tensor<f32>) -> tensor<1x128xf32>
    %7 = "tfl.pseudo_const"() <{value = dense<[1, 128, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %8 = "tfl.reshape"(%6, %7) : (tensor<1x128xf32>, tensor<3xi32>) -> tensor<1x128x1xf32>
    %9 = "tfl.rsqrt"(%8) : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    %10 = tfl.mul(%arg0, %9) <{fused_activation_function = "NONE"}> : (tensor<1x128x2304xf32>, tensor<1x128x1xf32>) -> tensor<1x128x2304xf32>
    %11 = tfl.mul(%10, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x128x2304xf32>, tensor<2304xf32>) -> tensor<1x128x2304xf32>
    return %11 : tensor<1x128x2304xf32>
  }
}