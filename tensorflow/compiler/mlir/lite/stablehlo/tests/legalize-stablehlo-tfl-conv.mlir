// RUN: odml-to-stablehlo-opt %s -stablehlo-tfl | FileCheck -dump-input always %s

module {
func.func @main(%arg0: tensor<8x8x1x207xf32>, %arg1: tensor<3x3x16x207xf32>) -> tensor<16x8x8x1xf32> {
  %0 = "stablehlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 2,
      input_feature_dimension = 3,
      input_spatial_dimensions = [0, 1],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 3,
      output_feature_dimension = 0,
      output_spatial_dimensions = [1, 2]
    >, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<1> : tensor<2x2xi64>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} :
       (tensor<8x8x1x207xf32>, tensor<3x3x16x207xf32>) -> tensor<16x8x8x1xf32>
  func.return %0 : tensor<16x8x8x1xf32>
}
}

// CHECK:      module {
// CHECK-NEXT:    func @main(%arg0: tensor<8x8x1x207xf32>, %arg1: tensor<3x3x16x207xf32>) -> tensor<16x8x8x1xf32> {
// CHECK-NEXT:    %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "stablehlo.convolution", custom_option = #tfl<const_bytes : "0x62617463685F67726F75705F636F756E740064696D656E73696F6E5F6E756D62657273000200010404020001040402010204040902031103020F03000D040428040428040428666561747572655F67726F75705F636F756E74006C68735F64696C6174696F6E0002010170616464696E67000401010101707265636973696F6E5F636F6E666967000744454641554C54000744454641554C540002120A7268735F64696C6174696F6E0002010177696E646F775F737472696465730002010108C0AF7C695A4E291A080108019801665C3526150428042C2C3C2C2C102401">} : (tensor<8x8x1x207xf32>, tensor<3x3x16x207xf32>) -> tensor<16x8x8x1xf32>
// CHECK-NEXT:    return %0 : tensor<16x8x8x1xf32>
// CHECK-NEXT:    }
// CHECK-NEXT: }
