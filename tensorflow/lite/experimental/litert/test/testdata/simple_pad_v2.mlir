module {
func.func @main(%input: tensor<1x112x112x64xf32>) -> tensor<1x114x114x64xf32> {
  %padding = "tfl.pseudo_const"() {
    value = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  } : () -> tensor<4x2xi32>
  %constant_values = "tfl.pseudo_const"() {
    value = dense<0.0> : tensor<f32>
  } : () -> tensor<f32>
  %output = "tfl.padv2"(%input, %padding, %constant_values) {
  }: (tensor<1x112x112x64xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<1x114x114x64xf32>
  return %output : tensor<1x114x114x64xf32>
}
}
