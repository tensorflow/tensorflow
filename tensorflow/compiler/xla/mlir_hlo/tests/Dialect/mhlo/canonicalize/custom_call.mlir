// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// CHECK-LABEL:@noeffect
func.func @noeffect(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NOT: custom_call
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "foo", has_side_effect = false} : (tensor<8xf32>) -> tensor<8xf32>
  func.return %arg0 : tensor<8xf32>
}

// CHECK-LABEL:@sideeffect
func.func @sideeffect(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: custom_call
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "foo", has_side_effect = true} : (tensor<8xf32>) -> tensor<8xf32>
  func.return %arg0 : tensor<8xf32>
}

// CHECK-LABEL:@defaulteffect
func.func @defaulteffect(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: custom_call
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "foo"} : (tensor<8xf32>) -> tensor<8xf32>
  func.return %arg0 : tensor<8xf32>
}

