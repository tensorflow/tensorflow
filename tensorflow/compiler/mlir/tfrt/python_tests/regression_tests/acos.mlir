builtin.func @test(%V__0: tensor<?x?xf32> { python_test_attrs.static_type = tensor<10x5xf32> }) -> tensor<?x?xf32> {
  %1 = "tf.Acos"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
