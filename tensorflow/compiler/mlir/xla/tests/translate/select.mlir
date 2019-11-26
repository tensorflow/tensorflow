// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: ENTRY %main
func @main(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: %[[ARG0:.*]] = pred[] parameter(0)
  // CHECK: %[[COND:.*]] = pred[2,3] broadcast(pred[] %Arg_0.1), dimensions={}
  // CHECK: %[[ARG1:.*]] = s32[2,3] parameter(1)
  // CHECK: %[[ARG2:.*]] = s32[2,3] parameter(2)

  // CHECK: ROOT %[[RES:.*]] = s32[2,3] select(pred[2,3] %[[COND]], s32[2,3] %[[ARG1]], s32[2,3] %[[ARG2]])
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) {name = "select.4"} : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

