module {
func.func @main(%arg0: tensor<1x216x288x48xf32>) -> tensor<1x216x288x48xf32> {
  %0 = "tfl.hard_swish"(%arg0) : (tensor<1x216x288x48xf32>) -> tensor<1x216x288x48xf32>
  return %0 : tensor<1x216x288x48xf32>
}
}