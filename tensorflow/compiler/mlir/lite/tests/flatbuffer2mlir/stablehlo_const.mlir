// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// test stablehlo roundtrip

module attributes {tfl.metadata = {"keep_stablehlo_constant" = "true"}} {
 func.func @main () -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
 }
}

//CHECK:module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {keep_stablehlo_constant = "true"}, tfl.schema_version = 3 : i32} {
//CHECK-NEXT: func.func @main() -> tensor<1x1x1x96xf32> attributes {tf.entry_function = {outputs = "stablehlo.constant"}} {
//CHECK-NEXT:  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x1x96xf32>
//CHECK-NEXT:  return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: }
//CHECK-NEXT:}