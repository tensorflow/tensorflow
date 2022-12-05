// RUN: tf-mhlo-tfl-opt %s -tfl-parse-mhlo-ops | FileCheck -dump-input always %s

module {
func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tfl.custom"(%arg0) {custom_code = "mhlo.rsqrt", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
}

// CHECK:       module
// CHECK-NEXT:  func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT:  %0 = mhlo.rsqrt %arg0 : tensor<2xf32>
// CHECK-NEXT:  return %0 : tensor<2xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
