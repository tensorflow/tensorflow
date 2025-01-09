module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "tfl.pseudo_const"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = tfl.mul %arg0, %0 {fused_activation_function = "NONE"} : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
  func.func @other(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "tfl.pseudo_const"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = tfl.mul %arg0, %0 {fused_activation_function = "NONE"} : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}