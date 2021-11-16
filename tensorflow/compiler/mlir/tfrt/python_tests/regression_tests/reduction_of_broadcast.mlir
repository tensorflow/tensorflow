builtin.func @test(%V__0: tensor<f32>, %V__1: tensor<2xi32>) -> tensor<?xf32> {
  %0 = "tf.BroadcastTo"(%V__0, %V__1) : (tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32>
  %dims1 = "tf.Const"() { value = dense<[0]> : tensor<1xi32> }: () -> tensor<1xi32>
  %1 = "tf.Max"(%0, %dims1) {keep_dims = false} : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
