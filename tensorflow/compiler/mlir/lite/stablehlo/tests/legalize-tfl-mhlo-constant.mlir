// RUN: tf-mhlo-tfl-opt %s -tfl-parse-mhlo-ops | FileCheck -dump-input always %s

module {
  func.func @main() -> tensor<1xi64> {
  %0 = "tfl.custom"() {custom_code = "mhlo.constant", custom_option = #tfl<const_bytes : "0x76616C75650001020109010101062C022401">} : () -> tensor<1xi64>
  func.return %0 : tensor<1xi64>
  }
}

// CHECK:  module {
// CHECK-NEXT:    func @main() -> tensor<1xi64> {
// CHECK-NEXT:    %0 = mhlo.constant dense<2> : tensor<1xi64>
// CHECK-NEXT:    return %0 : tensor<1xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    }
