// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: main
// CHECK: %[[ARG0:.*]] = (f32[], s32[]) parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = f32[] get-tuple-element((f32[], s32[]) %[[ARG0]]), index=0
