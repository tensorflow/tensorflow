builtin.func @test(%V__0 : tensor<?x?xf32> { python_test_attrs.static_type = tensor<1x16xf32> }, %V__1 : tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<15x75x16xf32> }) -> tensor<?x?x?xf32> {
  %0 = "tf.FloorMod"(%V__0, %V__1) : (tensor<?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
