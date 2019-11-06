// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<2xf32>) -> tensor<1x2xf32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// CHECK: ENTRY %main
// CHECK: %[[ARG0:.*]] = f32[2] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = f32[1,2] reshape(f32[2] %[[ARG0]])
