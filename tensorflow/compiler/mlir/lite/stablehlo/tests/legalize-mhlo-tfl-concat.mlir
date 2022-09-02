// RUN: tf-mhlo-tfl-opt %s -mhlo-tfl | FileCheck %s

module {
func.func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  %1 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  func.return %1 : tensor<6x3xf32>
}
}

// CHECK:      module {
// CHECK-NEXT:    func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
// CHECK-NEXT:      %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "mhlo.concatenate", custom_option = #tfl<const_bytes : "0x64696D656E73696F6E00010B0101010004022401">} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
// CHECK-NEXT:      return %0 : tensor<6x3xf32>
// CHECK-NEXT:    }
// CHECK-NEXT: }



