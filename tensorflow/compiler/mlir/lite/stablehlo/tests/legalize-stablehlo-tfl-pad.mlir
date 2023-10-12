// RUN: odml-to-stablehlo-opt %s -stablehlo-tfl | FileCheck %s

module {
func.func @main(%arg0: tensor<8x128xf32>, %arg1: tensor<f32>) -> tensor<11x131xf32> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[1, 0]> : tensor<2xi64>,
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<8x128xf32>, tensor<f32>) -> tensor<11x131xf32>
  func.return %0 : tensor<11x131xf32>
}
}

// CHECK:       module {
// CHECK-NEXT:    func @main(%arg0: tensor<8x128xf32>, %arg1: tensor<f32>) -> tensor<11x131xf32> {
// CHECK-NEXT:      %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "stablehlo.pad", custom_option = #tfl<const_bytes : "0x656467655F70616464696E675F6869676800020203656467655F70616464696E675F6C6F7700020100696E746572696F725F70616464696E6700020000033E2A17030103311E0B2C2C2C062401">} : (tensor<8x128xf32>, tensor<f32>) -> tensor<11x131xf32>
// CHECK-NEXT:      return %0 : tensor<11x131xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
