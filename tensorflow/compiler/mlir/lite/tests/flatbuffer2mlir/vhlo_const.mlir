// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir --disable-vhlo-to-stablehlo=true - -o - | FileCheck %s
// test stablehlo roundtrip

module attributes {tfl.metadata = {"keep_stablehlo_constant" = "true"}} {
 func.func @main () -> tensor<1x1x1x96xf32> {
  %0 = "vhlo.constant_v1"() <{value = #vhlo.tensor_v1<dense<0.000000e+00> : tensor<f32>>}> : () -> tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
 }
}

//CHECK: func.func @main() -> tensor<1x1x1x96xf32> attributes {tf.entry_function = {outputs = "vhlo.constant_v1"}} {
//CHECK-NEXT:  %0 = "vhlo.constant_v1"() <{value = #vhlo.tensor_v1<dense<0.000000e+00> : tensor<1x1x1x96xf32>>}> : () -> tensor<1x1x1x96xf32>
//CHECK-NEXT:  return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT: }