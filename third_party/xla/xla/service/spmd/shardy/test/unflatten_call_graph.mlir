// RUN: sdy_opt %s -split-input-file -xla-sdy-unflatten-call-graph | FileCheck %s

// CHECK-LABEL: func @singleton(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func @singleton(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}
