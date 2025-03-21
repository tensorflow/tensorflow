module {
func.func @main(%input: tensor<1x224x224x3xf32>) -> tensor<1x230x230x3xf32> {
  %padding = "tfl.pseudo_const"() {
    value = dense<[[0, 0], [3, 3], [3, 3], [0, 0]]> : tensor<4x2xi32>
  } : () -> tensor<4x2xi32>
  %output = "tfl.pad"(%input, %padding) {
  }: (tensor<1x224x224x3xf32>, tensor<4x2xi32>) -> tensor<1x230x230x3xf32>
  return %output : tensor<1x230x230x3xf32>
}
}
