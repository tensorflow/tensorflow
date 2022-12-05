//RUN: tf_tfl_translate --enable-stablehlo-conversion --input-mlir %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s


module {
func.func @tfInplaceUpdate(%arg0: tensor<2x1x2xf32>) -> tensor<2x1x2xf32> {
  %1 = arith.constant dense<1> : tensor<1xi32>
  %2 = arith.constant dense<2.0> : tensor<1x1x2xf32>
  %3 = "tf.InplaceUpdate"(%arg0, %1, %2) {device = ""}
    : (tensor<2x1x2xf32>, tensor<1xi32>, tensor<1x1x2xf32>) -> tensor<2x1x2xf32>
  func.return %3 : tensor<2x1x2xf32>
}
}

//CHECK: module attributes {tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
//CHECK-NEXT:  func.func @main(%arg0: tensor<2x1x2xf32>) -> tensor<2x1x2xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.custom3"}} {
//CHECK-NEXT:    %[[cst0:.*]] = "tfl.custom"() {custom_code = "mhlo.constant", custom_option = #tfl<const_bytes : "0x76616C75650001010109010101062C022401">} : () -> tensor<i32>
//CHECK-NEXT:    %[[cst1:.*]] = "tfl.custom"() {custom_code = "mhlo.constant", custom_option = #tfl<const_bytes : "0x76616C75650001000109010101062C022401">} : () -> tensor<i32>
//CHECK-NEXT:    %[[cst2:.*]] = "tfl.custom"() {custom_code = "mhlo.constant", custom_option = #tfl<const_bytes : "0x76616C756500000002000000000000400000004001150101010D36022401">} : () -> tensor<1x1x2xf32>
//CHECK-NEXT:    %[[dus:.*]] = "tfl.custom"(%arg0, %[[cst2]], %[[cst0]], %[[cst1]], %[[cst1]]) {custom_code = "mhlo.dynamic_update_slice", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2x1x2xf32>, tensor<1x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x1x2xf32>
//CHECK-NEXT:    return %[[dus]] : tensor<2x1x2xf32>
//CHECK-NEXT:  }
//CHECK-NEXT:}
