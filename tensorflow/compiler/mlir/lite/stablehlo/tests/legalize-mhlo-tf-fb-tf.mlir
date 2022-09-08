// RUN: tf-mhlo-tfl-opt %s -mhlo-tfl | flatbuffer_translate -mlir-to-tflite-flatbuffer - -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

module {
func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = mhlo.add %arg0, %arg0 : tensor<2xi32>
  %1 = mhlo.subtract %0, %arg0 : tensor<2xi32>
  func.return %1 : tensor<2xi32>
}
}

// CHECK:     module attributes {tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
// CHECK-NEXT:  func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.custom1"}} {
// CHECK-NEXT:    %0 = "tfl.custom"(%arg0, %arg0) {custom_code = "mhlo.add", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK-NEXT:    %1 = "tfl.custom"(%0, %arg0) {custom_code = "mhlo.subtract", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK-NEXT:    return %1 : tensor<2xi32>
// CHECK-NEXT:  }
// CHECK-NEXT:}
