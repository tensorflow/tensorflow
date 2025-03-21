module {
func.func @main(%params: tensor<128x1x14x14xf32>) -> tensor<4x1x14x14xf32> {
  %indices = "tfl.pseudo_const"() {
    value = dense<[[0], [2], [4], [6]]> : tensor<4x1xi32>
  } : () -> tensor<4x1xi32>
  %output = "tfl.gather_nd"(%params, %indices) {
  }: (tensor<128x1x14x14xf32>, tensor<4x1xi32>) -> tensor<4x1x14x14xf32>
  return %output : tensor<4x1x14x14xf32>
}
}
