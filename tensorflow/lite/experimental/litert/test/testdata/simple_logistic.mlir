module {
func.func @main(%input: tensor<1x256x128x3xf32>) -> tensor<1x256x128x3xf32> {
  %output = "tfl.logistic"(%input) {
  }: (tensor<1x256x128x3xf32>) -> tensor<1x256x128x3xf32>
  return %output : tensor<1x256x128x3xf32>
}
}
