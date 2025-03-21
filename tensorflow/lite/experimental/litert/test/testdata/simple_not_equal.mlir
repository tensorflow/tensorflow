module {
func.func @main(%input1: tensor<1x256x128x3xf32>, %input2: tensor<1x256x128x3xf32>) -> tensor<1x256x128x3xi1> {
  %output = "tfl.not_equal"(%input1, %input2) {
  }: (tensor<1x256x128x3xf32>, tensor<1x256x128x3xf32>) -> tensor<1x256x128x3xi1>
  return %output : tensor<1x256x128x3xi1>
}
}
