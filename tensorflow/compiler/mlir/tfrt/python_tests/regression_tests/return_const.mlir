func.func @test() -> tensor<2xi32> {
  %0 = "tf.Const"() { value = dense<[38, 33]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}
