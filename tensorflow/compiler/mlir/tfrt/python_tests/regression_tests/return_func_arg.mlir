func.func @test(%V__0 : tensor<?x?x?x?xf32> { python_test_attrs.static_type = tensor<13x1x1x76xf32> }) -> tensor<?x?x?x?xf32> {
  func.return %V__0 : tensor<?x?x?x?xf32>
}
