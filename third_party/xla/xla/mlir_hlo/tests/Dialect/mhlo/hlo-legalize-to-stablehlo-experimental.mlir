// RUN: mlir-hlo-opt --hlo-legalize-to-stablehlo --mlir-print-op-generic --split-input-file --verify-diagnostics %s
// RUN: mlir-hlo-opt --hlo-legalize-to-stablehlo=allow-experimental-features --split-input-file --mlir-print-op-generic %s | FileCheck %s

// This test file runs both FileCheck and diagnostic check. These tests all
// error when the experimental flag is disabled, and pass when it is enabled.

// CHECK-LABEL: "op_all_to_all_tuple"
func.func @op_all_to_all_tuple(%arg0: tensor<128x4xf32>, %arg1: tensor<128x4xf32>) -> (tensor<128x4xf32>, tensor<128x4xf32>) {
  //               CHECK: "stablehlo.custom_call"(%arg0, %arg1) <{call_target_name = "mhlo.all_to_all"}> {
  // CHECK-SAME{LITERAL}:    mhlo.attributes = {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}
  //          CHECK-SAME: } : (tensor<128x4xf32>, tensor<128x4xf32>)
  // expected-error@+1 {{failed to legalize operation 'mhlo.all_to_all' that was explicitly marked illegal}}
  %0:2 = "mhlo.all_to_all"(%arg0, %arg1) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<128x4xf32>, tensor<128x4xf32>) -> (tensor<128x4xf32>, tensor<128x4xf32>)
  return %0#0, %0#1 : tensor<128x4xf32>, tensor<128x4xf32>
}

