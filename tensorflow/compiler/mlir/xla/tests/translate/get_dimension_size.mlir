// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg: tensor<4x2xf32>) -> tensor<i32> {
  %0 = "xla_hlo.get_dimension_size"(%arg) {dimension = 1 : i32} : (tensor<4x2xf32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: ENTRY
// CHECK: [[ARG:%.*]] = f32[4,2] parameter(0)
// CHECK: s32[] get-dimension-size(f32[4,2] [[ARG]]), dimensions={1}
