// RUN: odml-to-stablehlo-opt %s -tfl-parse-stablehlo-ops | FileCheck %s

module {
func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.custom1"}} {
  %0 = "tfl.custom"(%arg0, %arg0) {custom_code = "stablehlo.add", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = "tfl.custom"(%0, %arg0) {custom_code = "stablehlo.subtract", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %1 : tensor<2xi32>
}
}

// CHECK:       module {
// CHECK-NEXT:  func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.custom1"}} {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg0 : tensor<2xi32>
// CHECK-NEXT:    %1 = stablehlo.subtract %0, %arg0 : tensor<2xi32>
// CHECK-NEXT:    return %1 : tensor<2xi32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
