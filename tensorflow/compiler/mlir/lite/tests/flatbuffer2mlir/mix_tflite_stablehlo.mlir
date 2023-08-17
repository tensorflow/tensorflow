// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// test stablehlo roundtrip

module {
func.func @main(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.logistic %arg0 : tensor<1x1x1x96xf32>
  %1 = "tfl.exp"(%0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> loc("exp")
  func.return %1 : tensor<1x1x1x96xf32>
}
}

// CHECK:module attributes {tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
// CHECK-NEXT: func.func @main(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "exp"}} {
// CHECK-NEXT:  %0 = stablehlo.logistic %arg0 : tensor<1x1x1x96xf32>
// CHECK-NEXT:  %1 = "tfl.exp"(%0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
// CHECK-NEXT:  return %1 : tensor<1x1x1x96xf32>
// CHECK-NEXT: }
// CHECK-NEXT:}