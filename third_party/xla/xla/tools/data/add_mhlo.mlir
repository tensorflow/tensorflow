func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = mhlo.add %arg0, %arg0 : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
