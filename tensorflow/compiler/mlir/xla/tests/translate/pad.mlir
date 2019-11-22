// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg: tensor<4x6xf32>, %pad: tensor<f32>) -> tensor<13x19xf32> {
  %0 = "xla_hlo.pad"(%arg, %pad) {edge_padding_high = dense<[4,5]> : tensor<2xi64>, edge_padding_low = dense<[2,3]> : tensor<2xi64>, interior_padding = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<13x19xf32>
  return %0 : tensor<13x19xf32>
}

// CHECK-LABEL:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[4,6] parameter(0)
// CHECK:  [[PADDING_VAL:%.*]] = f32[] parameter(1)
// CHECK-LABEL:  ROOT
// CHECK-SAME:  f32[13,19] pad(f32[4,6] [[ARG]], f32[] [[PADDING_VAL]]), padding=2_4_1x3_5_1
