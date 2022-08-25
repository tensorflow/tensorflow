// RUN: tf-mhlo-tfl-opt %s -mhlo-tfl | FileCheck %s

module {
func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = mhlo.maximum %arg0, %arg0 : tensor<2xi32>
  func.return %0 : tensor<2xi32>
}
}

// CHECK:       module {
// CHECK-NEXT:    func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
// CHECK-NEXT:      %0 = "tfl.custom"(%arg0, %arg0) {custom_code = "mhlo.maximum", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK-NEXT:      return %0 : tensor<2xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
