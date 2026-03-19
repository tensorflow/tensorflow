// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Ensure "quantfork.stats" roundtrip exactly

func.func @main(%arg0: tensor<1x512x672x8xf32>) -> tensor<1x512x672x8xf32> {
// CHECK-LABEL: @main
// CHECK: %[[RES0:.*]] = "quantfork.stats"(%arg0) <{layerStats = dense<[0.000000e+00, 2.550000e+02]> : tensor<2xf32>}> : (tensor<1x512x672x8xf32>) -> tensor<1x512x672x8xf32>

  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 2.550000e+02]> : tensor<2xf32>} : (tensor<1x512x672x8xf32>) -> tensor<1x512x672x8xf32>
  func.return %0 : tensor<1x512x672x8xf32>
}
