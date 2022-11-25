// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_handle_data
// CHECK: -> (tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>)
tfg.func @test_handle_data(%arg0: tensor<*xi32>)
    -> (tensor<*x!tf_type.resource> {tfg.handle_data = [tensor<4xf32>, tensor<2xi32>]}) {
  // CHECK: %[[A:.*]], %{{.*}} = A(%{{.*}}) : (tensor<*xi32>) -> (tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>)
  %A, %ctl = A(%arg0) : (tensor<*xi32>) -> (tensor<*x!tf_type.resource>)
  // CHECK: return(%[[A]]) : tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>
  return(%A) : tensor<*x!tf_type.resource>
}

// CHECK-LABEL: tfg.func @test_handle_data_arg_type(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>)
// CHECK: -> (tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>)
tfg.func @test_handle_data_arg_type(%arg0: tensor<*x!tf_type.resource>)
    -> (tensor<*x!tf_type.resource> {tfg.handle_data = [tensor<4xf32>, tensor<2xi32>]}) {
  // CHECK: return(%[[ARG0]]) : tensor<*x!tf_type.resource<tensor<4xf32>, tensor<2xi32>>>
  return(%arg0) : tensor<*x!tf_type.resource>
}
