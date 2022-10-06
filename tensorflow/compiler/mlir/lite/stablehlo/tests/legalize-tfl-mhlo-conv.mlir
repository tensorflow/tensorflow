// RUN: tf-mhlo-tfl-opt %s -tfl-parse-mhlo-ops | FileCheck -dump-input always %s

module {
  func.func @main(%arg0: tensor<8x8x1x207xf32>, %arg1: tensor<3x3x16x207xf32>) -> tensor<16x8x8x1xf32> {
  %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "mhlo.convolution", custom_option = #tfl<const_bytes : "0x62617463685F67726F75705F636F756E740064696D656E73696F6E5F6E756D62657273000200010404020001040402010204040902031103020F03000D040428040428040428666561747572655F67726F75705F636F756E74006C68735F64696C6174696F6E0002010170616464696E67000401010101707265636973696F6E5F636F6E666967000744454641554C54000744454641554C540002120A7268735F64696C6174696F6E0002010177696E646F775F737472696465730002010108C0AF7C695A4E291A080108019801665C3526150428042C2C3C2C2C102401">} : (tensor<8x8x1x207xf32>, tensor<3x3x16x207xf32>) -> tensor<16x8x8x1xf32>
  func.return %0 : tensor<16x8x8x1xf32>
  }
}


// CHECK:      module {
// CHECK:      func @main(%arg0: tensor<8x8x1x207xf32>, %arg1: tensor<3x3x16x207xf32>) -> tensor<16x8x8x1xf32> {
// CHECK:      %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [0, 1, b, f]x[0, 1, o, i]->[f, 0, 1, b], window = {stride = [1, 1], pad = [1, 1, 1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<8x8x1x207xf32>, tensor<3x3x16x207xf32>) -> tensor<16x8x8x1xf32>
// CHECK:      return %0 : tensor<16x8x8x1xf32>
