module @foo {
  func.func public @main(%inputs : tensor<1x4x4x2xf32>, %weights : tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32> {
    %res = "mhlo.convolution"(%inputs, %weights) {
          batch_group_count = 1 : i64,
          dimension_numbers = #mhlo.conv<raw
            input_batch_dimension = 0,
            input_feature_dimension = 3,
            input_spatial_dimensions = [1, 2],
            kernel_input_feature_dimension = 2,
            kernel_output_feature_dimension = 3,
            kernel_spatial_dimensions = [0, 1],
            output_batch_dimension = 0,
            output_feature_dimension = 3,
            output_spatial_dimensions = [1, 2]
          >,
          feature_group_count = 1 : i64,
          rhs_dilation = dense<1> : tensor<2xi64>,
          window_strides = dense<1> : tensor<2xi64>} : (tensor<1x4x4x2xf32>, tensor<3x2x2x1xf32>) -> tensor<1x2x3x1xf32>
    return %res : tensor<1x2x3x1xf32>
  }
}