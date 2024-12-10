module {
func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %1 = tfl.mul %0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %2 = tfl.add %1, %1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %3 = tfl.mul %2, %2 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
}
}