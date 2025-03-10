module {
func.func @main(%arg0: tensor<1x32x32x192xf32>) -> tensor<1x32x32x192xf32> {
  %0 = "tfl.leaky_relu"(%arg0) <{alpha = 2.000000e-01 : f32}> : (tensor<1x32x32x192xf32>) -> tensor<1x32x32x192xf32>
  return %0 : tensor<1x32x32x192xf32>
}
}