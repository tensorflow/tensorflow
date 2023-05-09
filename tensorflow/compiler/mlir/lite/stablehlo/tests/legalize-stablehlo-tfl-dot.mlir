// RUN: odml-to-stablehlo-opt %s -stablehlo-tfl | FileCheck %s

module {
func.func @main(%arg0: tensor<72x2048xf32>, %arg1: tensor<2048x512xf32>) -> tensor<72x512xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [1, 2],
      lhs_contracting_dimensions = [0, 1],
      rhs_contracting_dimensions = [1, 2]
    >} :
       (tensor<72x2048xf32>, tensor<2048x512xf32>) -> tensor<72x512xf32>
  func.return %0 : tensor<72x512xf32>
}
}

// CHECK:      module {
// CHECK-NEXT:    func.func @main(%arg0: tensor<72x2048xf32>, %arg1: tensor<2048x512xf32>) -> tensor<72x512xf32> {
// CHECK-NEXT:    %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "stablehlo.dot", custom_option = #tfl<const_bytes : "0x64696D656E73696F6E5F6E756D626572730002000104040201020404020001040402010204040414100C082828282801300101010D28022401">} : (tensor<72x2048xf32>, tensor<2048x512xf32>) -> tensor<72x512xf32>
// CHECK-NEXT:    return %0 : tensor<72x512xf32>
// CHECK-NEXT:    }
// CHECK-NEXT: }
