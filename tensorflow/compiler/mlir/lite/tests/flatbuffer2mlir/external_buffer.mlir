// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

module {
  func.func public @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "tfl.external_const"() <{external_buffer = #tfl.external_buffer<group_name = "test.bin", offset = 0, length = 13, packing = "unpacked">}> : () -> tensor<2x2xf32>
    %1 = tfl.add %arg0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: @main
// CHECK:      %0 = "tfl.external_const"() <{external_buffer = #tfl.external_buffer<group_name = "test.bin", offset = 0, length = 13, packing = "unpacked">}>
// CHECK-NEXT: %1 = tfl.add %arg0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK-NEXT: return %1
