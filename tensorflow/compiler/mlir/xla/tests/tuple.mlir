// RUN: tf-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: func @fold_access
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @fold_access(%arg : tensor<i32>) -> tensor<i32> {
  // CHECK-NEXT: return [[ARG]]
  %tuple = "xla_hlo.tuple"(%arg) : (tensor<i32>) -> tuple<tensor<i32>>
  %element = "xla_hlo.get_tuple_element"(%tuple) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
  return %element : tensor<i32>
}
