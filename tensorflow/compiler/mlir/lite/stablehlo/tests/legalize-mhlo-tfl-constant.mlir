// RUN: tf-mhlo-tfl-opt %s -mhlo-tfl | FileCheck %s

module {
func.func @main() -> tensor<2xf32> {
  %0 = mhlo.constant dense<2> : tensor<i32>
  %1 = mhlo.constant dense<[10.0, 11.0]> : tensor<2xf32>
  func.return %1 : tensor<2xf32>
}
}

// CHECK:       module {
// CHECK-NEXT:    func.func @main() -> tensor<2xf32> {
// CHECK-NEXT:      %0 = "tfl.custom"() {custom_code = "mhlo.constant", custom_option = #tfl<const_bytes : "0x76616C75650001020109010101062C022401">} : () -> tensor<i32>
// CHECK-NEXT:      %1 = "tfl.custom"() {custom_code = "mhlo.constant", custom_option = #tfl<const_bytes : "0x76616C756500000002000000000020410000304101150101010D36022401">} : () -> tensor<2xf32>
// CHECK-NEXT:      return %1 : tensor<2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
