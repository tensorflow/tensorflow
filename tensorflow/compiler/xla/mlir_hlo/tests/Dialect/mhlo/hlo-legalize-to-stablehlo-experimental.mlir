// RUN: mlir-hlo-opt --hlo-legalize-to-stablehlo --mlir-print-op-generic --split-input-file --verify-diagnostics %s
// RUN: mlir-hlo-opt --hlo-legalize-to-stablehlo=allow-experimental-features --mlir-print-op-generic %s | FileCheck %s

// This test file runs both FileCheck and diagnostic check. These tests all
// error when the experimental flag is disabled, and pass when it is enabled.

// CHECK-LABEL: "op_all_to_all_tuple"
func.func @op_all_to_all_tuple(%arg0: tensor<128x4xf32>, %arg1: tensor<128x4xf32>) -> (tensor<128x4xf32>, tensor<128x4xf32>) {
  //               CHECK: "stablehlo.custom_call"(%arg0, %arg1) {
  //          CHECK-SAME:    call_target_name = "mhlo.all_to_all"
  // CHECK-SAME{LITERAL}:    mhlo.attributes = {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}
  //          CHECK-SAME: } : (tensor<128x4xf32>, tensor<128x4xf32>)
  // expected-error@+1 {{failed to legalize operation 'mhlo.all_to_all' that was explicitly marked illegal}}
  %0:2 = "mhlo.all_to_all"(%arg0, %arg1) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<128x4xf32>, tensor<128x4xf32>) -> (tensor<128x4xf32>, tensor<128x4xf32>)
  return %0#0, %0#1 : tensor<128x4xf32>, tensor<128x4xf32>
}

// -----

// CHECK-LABEL: "op_custom_call_api_version_typed_ffi"
func.func @op_custom_call_api_version_typed_ffi(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "stablehlo.custom_call"(%arg0) {
  // CHECK-SAME:   call_target_name = "mhlo.custom_call"
  // CHECK-SAME:   mhlo.attributes = {api_version = 4 : i32, backend_config = {foo = "bar"}, call_target_name = "foo"}
  // CHECK-SAME: } : (tensor<f32>) -> tensor<f32>
  // expected-error@+1 {{failed to legalize operation 'mhlo.custom_call' that was explicitly marked illegal}}
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo",
    backend_config = {foo = "bar"},
    api_version = 4 : i32
  } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "attr_precision_packed_nibble"
func.func @attr_precision_packed_nibble(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  //      CHECK: "stablehlo.custom_call"(%arg0, %arg1) {
  // CHECK-SAME:    call_target_name = "mhlo.dot"
  // CHECK-SAME:    mhlo.attributes = {precision_config = ["PACKED_NIBBLE"]}
  // CHECK-SAME: } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  // expected-error@+1 {{failed to legalize operation 'mhlo.dot' that was explicitly marked illegal}}
  %0 = "mhlo.dot"(%arg0, %arg1) {
    precision_config = [#mhlo<precision PACKED_NIBBLE>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
