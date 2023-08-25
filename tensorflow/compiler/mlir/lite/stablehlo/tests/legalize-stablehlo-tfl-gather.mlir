// RUN: odml-to-stablehlo-opt %s -stablehlo-tfl | FileCheck %s

module {
func.func @main(%arg0: tensor<1x128x256xf32>, %arg1: tensor<30x1x2xi32>) -> tensor<30x1x256xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 256]> : tensor<3xi64>} :
       (tensor<1x128x256xf32>, tensor<30x1x2xi32>) -> tensor<30x1x256xf32>
  func.return %0 : tensor<30x1x256xf32>
}
}

// CHECK:      module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<1x128x256xf32>, %arg1: tensor<30x1x2xi32>) -> tensor<30x1x256xf32> {
// CHECK-NEXT:     %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "stablehlo.gather", custom_option = #tfl<const_bytes : "0x64696D656E73696F6E5F6E756D626572730001020402000104040200010404040D0B070228282804696E64696365735F6172655F736F7274656400736C6963655F73697A65730000030001000100000103512A1803010337000F28042D062401">} : (tensor<1x128x256xf32>, tensor<30x1x2xi32>) -> tensor<30x1x256xf32>
// CHECK-NEXT:     return %0 : tensor<30x1x256xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
