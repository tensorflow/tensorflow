// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// test stablehlo roundtrip

module {
func.func @main(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.logistic %arg0 : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}
}

// CHECK:module attributes {tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
// CHECK: func.func @main(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "stablehlo.logistic"}} {
// CHECK:  %0 = stablehlo.logistic %arg0 : tensor<1x1x1x96xf32>
// CHECK:  return %0 : tensor<1x1x1x96xf32>
// CHECK: }
// CHECK:}