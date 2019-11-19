// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "xla_hlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK: ENTRY %main
// CHECK: %[[ARG:.*]] = s32[2] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = f32[2] convert(s32[2] %[[ARG]])
