// RUN: not mlir-bisect %s --hlo-snapshot=%s.pb \
// RUN: --pass-pipeline="builtin.module(test-break-linalg-transpose)" \
// RUN: | FileCheck %s

func.func @main(%a: tensor<3x1xi32>, %b: tensor<3x1xi32>) -> tensor<3x1xi32> {
  return %a : tensor<3x1xi32>
}

// CHECK: initial module
// CHECK: func @main() -> tensor<3x1xi32> {
// CHECK{LITERAL}: arith.constant dense<[[2], [-4], [5]]> : tensor<3x1xi32>
// CHECK{LITERAL}: arith.constant dense<[[0], [7], [-5]]> : tensor<3x1xi32>
