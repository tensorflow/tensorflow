builtin.func @test(%V__0 : tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<90x46x74xf32> }, %V__1 : tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<90x1x74xf32> }) -> tensor<?x?x?xf32> {
  %0 = "tf.Const"() { value = dense<[90, 46, 74]> : tensor<3xi32> } : () -> tensor<3xi32>
  %1 = "tf.BroadcastTo"(%V__0, %0) : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %2 = "tf.FloorDiv"(%1, %V__1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %2 : tensor<?x?x?xf32>
}
