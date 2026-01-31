// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Ensure cast with bfloat16 roundtrip exactly

func.func @main(tensor<4x5xbf16>) -> tensor<4x5xbf16> {
^bb0(%arg0: tensor<4x5xbf16>):
  // CHECK-LABEL: @main
  // CHECK:  (tensor<4x5xbf16>) -> tensor<4x5xf32>
  // CHECK-NEXT:  (tensor<4x5xf32>) -> tensor<4x5xbf16>
  %0 = "tfl.cast" (%arg0) : (tensor<4x5xbf16>) -> tensor<4x5xf32> loc("cast1")
  %1 = "tfl.cast" (%0) : (tensor<4x5xf32>) -> tensor<4x5xbf16> loc("cast2")
  func.return %1 : tensor<4x5xbf16>
}
