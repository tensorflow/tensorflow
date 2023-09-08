// RUN: odml-to-stablehlo-opt %s -tfl-parse-stablehlo-ops | FileCheck -dump-input always %s

module {
func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tfl.custom"(%arg0) {custom_code = "stablehlo.rsqrt", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
}

// CHECK:       module
// CHECK-NEXT:  func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT:  %0 = stablehlo.rsqrt %arg0 : tensor<2xf32>
// CHECK-NEXT:  return %0 : tensor<2xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
