// RUN: mlir-hlo-opt --hlo-legalize-to-stablehlo --mlir-print-op-generic --split-input-file --verify-diagnostics %s
// RUN: mlir-hlo-opt --hlo-legalize-to-stablehlo=allow-experimental-features --split-input-file --mlir-print-op-generic %s | FileCheck %s

// This test file runs both FileCheck and diagnostic check. These tests all
// error when the experimental flag is disabled, and pass when it is enabled.

// CHECK-LABEL: "op_all_reduce_tuple"
func.func @op_all_reduce_tuple(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> (tensor<8xf32>, tensor<f32>) {
  //               CHECK: "stablehlo.custom_call"(%[[ARG0:.*]], %[[ARG1:.*]]) <{
  //          CHECK-SAME:  call_target_name = "mhlo.all_reduce", called_computations = [@all_reduce]
  //          CHECK-SAME: }> {
  // CHECK-SAME{LITERAL}:    mhlo.attributes = {replica_groups = dense<> : tensor<0x0xi64>}
  //          CHECK-SAME: } : (tensor<8xf32>, tensor<f32>) -> (tensor<8xf32>, tensor<f32>)
  //               CHECK: func.func
  //          CHECK-SAME: sym_name = "all_reduce"
  //               CHECK: ^bb0(%[[REDUCE_ARG0:.*]]: tensor<f32>, %[[REDUCE_ARG1:.*]]: tensor<f32>):
  //          CHECK-NEXT: %[[ADD:.*]] = "stablehlo.add"(%[[REDUCE_ARG0]], %[[REDUCE_ARG1]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  //          CHECK-NEXT: "stablehlo.return"(%[[ADD]]) : (tensor<f32>) -> ()
  // expected-error@+1 {{failed to legalize operation 'mhlo.all_reduce' that was explicitly marked illegal}}
  %0:2 = "mhlo.all_reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = mhlo.add %arg2, %arg3 : tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {replica_groups = dense<> : tensor<0x0xi64>} : (tensor<8xf32>, tensor<f32>) -> (tensor<8xf32>, tensor<f32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<f32>
}

// -----

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

// -----

// CHECK-LABEL: "attr_precision_packed_nibble"
func.func @attr_precision_packed_nibble(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  //      CHECK: "stablehlo.custom_call"(%arg0, %arg1) <{call_target_name = "mhlo.dot"}> {
  // CHECK-SAME:    mhlo.attributes = {precision_config = ["PACKED_NIBBLE"]}
  // CHECK-SAME: } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  // expected-error@+1 {{failed to legalize operation 'mhlo.dot' that was explicitly marked illegal}}
  %0 = "mhlo.dot"(%arg0, %arg1) {
    precision_config = [#mhlo<precision PACKED_NIBBLE>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
