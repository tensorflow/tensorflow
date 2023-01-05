module @foo {
  func.func public @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #mhlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
    } : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    func.return %0 : tensor<3x3xf32>
  }
}