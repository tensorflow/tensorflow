// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<f32>, %arg1 : tensor<i32>) -> tuple<tensor<f32>, tensor<i32>> {
  %result = "xla_hlo.tuple"(%arg0, %arg1) {} : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>>
  return %result : tuple<tensor<f32>, tensor<i32>>
}

// CHECK-LABEL: main
// CHECK: %[[ARG0:.*]] = f32[] parameter(0)
// CHECK: %[[ARG1:.*]] = s32[] parameter(1)
// CHECK: ROOT %[[RESULT:.*]] = (f32[], s32[]) tuple(f32[] %[[ARG0]], s32[] %[[ARG1]])
