builtin.func @test() -> tensor<2xi32> {
  %0 = "tf.Const"() { value = dense<[38, 33]> : tensor<2xi32> } : () -> tensor<2xi32>
  return %0 : tensor<2xi32>
}
