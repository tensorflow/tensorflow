// RUN: odml-to-stablehlo-opt %s -stablehlo-tfl | FileCheck %s

module {
func.func @main(%arg0: tensor<1x2xi32>) -> tensor<1x2x2xi32> {
  %0= "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x2xi32>
  func.return %0 : tensor<1x2x2xi32>
}
}

// CHECK:       module {
// CHECK-NEXT:    func @main(%arg0: tensor<1x2xi32>) -> tensor<1x2x2xi32> {
// CHECK-NEXT:      %0 = "tfl.custom"(%arg0) {custom_code = "stablehlo.broadcast_in_dim", custom_option = #tfl<const_bytes : "0x62726F6164636173745F64696D656E73696F6E73000201020119010101072C022401">} : (tensor<1x2xi32>) -> tensor<1x2x2xi32>
// CHECK-NEXT:      return %0 : tensor<1x2x2xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
