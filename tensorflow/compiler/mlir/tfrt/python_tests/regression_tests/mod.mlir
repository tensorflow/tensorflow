builtin.func @test(%V__0: tensor<?x?xf32> { python_test_attrs.static_type = tensor<5x2xf32> }) -> tensor<?x?xf32> {
  %2 = "tf.Mod"(%V__0, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
