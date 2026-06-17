// RUN: tf-tfrt-opt -tfrt-remove-device-attribute %s | FileCheck %s

func.func @test(%arg0: !tfrt.chain, %arg1: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  %0 = corert.get_op_handler %arg0 "cpu"
  // CHECK: %[[RESULT:.*]] = corert.executeop(%[[ARG_0:.*]]) "tf.MatMul"(%[[ARG_1:.*]], %[[ARG_1]]) {T = f32, transpose_a = false, transpose_b = false} : 1
  %1 = corert.executeop(%0) "tf.MatMul"(%arg1, %arg1) {T = f32, device = "cpu", transpose_a = false, transpose_b = false} : 1
  tfrt.return %arg0, %1 : !tfrt.chain, !corert.tensorhandle
}
