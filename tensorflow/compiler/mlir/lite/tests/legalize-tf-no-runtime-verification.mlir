// RUN: tf-opt %s -tfl-prepare-tf -tfl-legalize-tf='run-tfl-runtime-verification=false'  -tfl-optimize | FileCheck %s

func.func @broadcast_to_bf16(%arg0: tensor<3xbf16>, %arg1: tensor<2xi64>) -> tensor<3x3xbf16> {
  %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<3xbf16>, tensor<2xi64>) -> tensor<3x3xbf16>
  func.return %0: tensor<3x3xbf16>

// CHECK-LABEL: broadcast_to_bf16
// CHECK:  [[CST:%.*]] = arith.constant dense<1.000000e+00> : tensor<3x3xbf16>
// CHECK:  [[MUL:%.*]] = tfl.mul(%arg0, [[CST]]) {fused_activation_function = "NONE"} : (tensor<3xbf16>, tensor<3x3xbf16>) -> tensor<3x3xbf16>
// CHECK:  return [[MUL]] : tensor<3x3xbf16>
}
