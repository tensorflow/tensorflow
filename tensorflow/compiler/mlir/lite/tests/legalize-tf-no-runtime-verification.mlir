// RUN: tf-opt %s -tfl-legalize-tf='run-tfl-runtime-verification=false' | FileCheck %s

func @broadcast_to_bf16(%arg0: tensor<3xbf16>, %arg1: tensor<2xi64>) -> tensor<3x3xbf16> {
  %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<3xbf16>, tensor<2xi64>) -> tensor<3x3xbf16>
  return %0: tensor<3x3xbf16>

// CHECK-LABEL: broadcast_to_bf16
// CHECK:  [[CST:%.*]] = constant dense<1.000000e+00> : tensor<bf16>
// CHECK:  [[FILL:%.*]] = "tfl.fill"(%arg1, [[CST]]) : (tensor<2xi64>, tensor<bf16>) -> tensor<3x3xbf16>
// CHECK:  [[MUL:%.*]] = "tfl.mul"(%arg0, [[FILL]]) {fused_activation_function = "NONE"} : (tensor<3xbf16>, tensor<3x3xbf16>) -> tensor<3x3xbf16>
// CHECK:  return [[MUL]] : tensor<3x3xbf16>
}
