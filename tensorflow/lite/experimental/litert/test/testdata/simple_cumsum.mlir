module {
func.func @main(%input: tensor<1x14x14xf32>) -> tensor<1x14x14xf32> {
  %axis = "tfl.pseudo_const"() {
    value = dense<1> : tensor<i32>
  } : () -> tensor<i32>
  %output = "tfl.cumsum"(%input, %axis) {
    exclusive = false,
    reverse = false
  } : (tensor<1x14x14xf32>, tensor<i32>) -> tensor<1x14x14xf32>
  return %output : tensor<1x14x14xf32>
}
}
