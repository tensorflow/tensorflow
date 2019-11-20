// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0:2 = call @callee(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
  return %0#0, %0#1 : tensor<4xi32>, tensor<4xi32>
}

func @callee(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "xla_hlo.mul"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0, %1 : tensor<4xi32>, tensor<4xi32>
}

// Get name of callee computation
// CHECK:  [[CALLEE:%.*]] ({{.*}}) -> ({{.*}}) {

// CHECK-LABEL:  ENTRY
// CHECK-SAME:  [[MAIN:%.*]] ([[ARG:.*]]: s32[4]) -> (s32[4], s32[4]) {
// CHECK:  %[[ARG]] = s32[4] parameter(0)
// CHECK:  [[CALL_OUT:%.*]] = (s32[4], s32[4]) call(s32[4] %[[ARG]], s32[4] %[[ARG]]), to_apply=[[CALLEE]]
// CHECK:  [[OUT_0:%.*]] = s32[4] get-tuple-element((s32[4], s32[4]) [[CALL_OUT]]), index=0
// CHECK:  [[OUT_1:%.*]] = s32[4] get-tuple-element((s32[4], s32[4]) [[CALL_OUT]]), index=1
// CHECK-LABEL:  ROOT
// CHECK-SAME:  (s32[4], s32[4]) tuple(s32[4] [[OUT_0]], s32[4] [[OUT_1]])
