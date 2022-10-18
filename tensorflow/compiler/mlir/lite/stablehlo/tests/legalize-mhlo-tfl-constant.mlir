// RUN: tf-mhlo-tfl-opt %s -mhlo-tfl | FileCheck %s

module {
func.func @main() -> tensor<i32> {
  %0 = mhlo.constant dense<2> : tensor<i32>
  func.return %0 : tensor<i32>
}
}

// CHECK:       module {
// CHECK-NEXT:    func @main() -> tensor<i32> {
// CHECK-NEXT:      %0 = "tfl.custom"() {custom_code = "mhlo.constant", custom_option = #tfl<const_bytes : "0x76616C75650001020109010101062C022401">} : () -> tensor<i32>
// CHECK-NEXT:      return %0 : tensor<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
