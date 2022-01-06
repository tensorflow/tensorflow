builtin.func @test(%V__0 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<36x80x87x24xf32> }) -> tensor<?x?x?x?xf32> {
  %0 = "tf.Asinh"(%V__0) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %dims1 = "tf.Const"() { value = dense<[2, 1, 3, 0]> : tensor<4xi32> } : () -> tensor<4xi32>
  %1 = "tf.Transpose"(%0, %dims1) : (tensor<?x?x?x?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
