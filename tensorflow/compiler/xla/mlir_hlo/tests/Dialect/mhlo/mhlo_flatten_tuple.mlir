// RUN: mlir-hlo-opt -mhlo-flatten-tuple %s | FileCheck %s

// CHECK-LABEL: @custom_call
// CHECK-SAME: %[[X:.*]]: tensor<6x3xf32>
// CHECK: %[[CALL:.+]]:2 = "mhlo.custom_call"(%[[X]]) {api_version = 2 : i32, call_target_name = "f"} : (tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>)
// CHECK: return %[[CALL]]#0, %[[CALL]]#1 : tensor<6xf32>, tensor<3xf32>
func.func @custom_call(%x: tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>) {
  %0 = "mhlo.custom_call"(%x) {api_version = 2 : i32, call_target_name = "f"}
    : (tensor<6x3xf32>) -> tuple<tensor<6xf32>, tensor<3xf32>>
  %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<6xf32>
  %2 = "mhlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<3xf32>
  return %1, %2 : tensor<6xf32>, tensor<3xf32>
}
