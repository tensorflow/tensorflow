// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_handle_data(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*x!tf_type.resource<tensor<4xi32>, tensor<8xf32>>>)
// CHECK-NEXT: -> (tensor<*x!tf_type.resource<tensor<4xi32>, tensor<8xf32>>>)
tfg.func @test_handle_data(%arg0: tensor<*x!tf_type.resource> {tfg.handle_data = [tensor<4xi32>, tensor<8xf32>]})
    -> (tensor<*x!tf_type.resource>) {
  // CHECK: return(%[[ARG0]]) : tensor<*x!tf_type.resource<tensor<4xi32>, tensor<8xf32>>>
  return(%arg0) : tensor<*x!tf_type.resource>
}
