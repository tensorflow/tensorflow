// RUN: mlir-hlo-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// CHECK-LABEL: func @single_operand
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @single_operand(%arg: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = "mhlo.concatenate"(%arg) {dimension = 0 : i64} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: return [[ARG]]
  func.return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @operand_with_unknown_rank
func.func @operand_with_unknown_rank(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-NEXT: mhlo.concatenate
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: return
  func.return %0 : tensor<*xf32>
}
