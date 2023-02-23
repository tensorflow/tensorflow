// RUN: odml-to-stablehlo-opt %s -stablehlo-tfl | FileCheck %s

module {
func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2xi1> {
  %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %1 = stablehlo.compare LT, %arg0, %arg1, TOTALORDER : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %2 = stablehlo.compare GT, %arg2, %arg3 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  func.return %2 : tensor<2xi1>
}
}

// CHECK:      module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2xi1> {
// CHECK-NEXT:     %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "stablehlo.compare", custom_option = #tfl<const_bytes : "0x636F6D70617269736F6E5F646972656374696F6E00024C5400011A0101010814022401">} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
// CHECK-NEXT:     %1 = "tfl.custom"(%arg0, %arg1) {custom_code = "stablehlo.compare", custom_option = #tfl<const_bytes : "0x636F6D706172655F74797065000A544F54414C4F5244455200636F6D70617269736F6E5F646972656374696F6E00024C540002331B0201022A0A1414042401">} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
// CHECK-NEXT:     %2 = "tfl.custom"(%arg2, %arg3) {custom_code = "stablehlo.compare", custom_option = #tfl<const_bytes : "0x636F6D70617269736F6E5F646972656374696F6E0002475400011A0101010814022401">} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT:     return %2 : tensor<2xi1>
// CHECK-NEXT:   }
// CHECK-NEXT: }
