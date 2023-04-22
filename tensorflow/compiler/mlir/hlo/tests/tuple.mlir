// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @fold_access
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @fold_access(%arg : tensor<i32>) -> tensor<i32> {
  // CHECK-NEXT: return [[ARG]]
  %tuple = "mhlo.tuple"(%arg) : (tensor<i32>) -> tuple<tensor<i32>>
  %element = "mhlo.get_tuple_element"(%tuple) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
  return %element : tensor<i32>
}
