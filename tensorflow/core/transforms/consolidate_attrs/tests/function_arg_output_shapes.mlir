// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_output_shapes(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2xi32> {tfg.regenerate_output_shapes},
// CHECK-NEXT: %[[ARG1:.*]]: tensor<*xi32>)
// CHECK-NEXT: -> (tensor<2xi32>)
tfg.func @test_output_shapes(%arg0: tensor<*xi32> {tf._output_shapes = [#tf_type.shape<2>]},
                             %arg1: tensor<*xi32>)
    -> (tensor<*xi32>) {
  // CHECK: return(%[[ARG0]]) : tensor<2xi32>
  return(%arg0) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_not_array_attr(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*xi32> {tfg.regenerate_output_shapes})
tfg.func @test_not_array_attr(%arg0: tensor<*xi32> {tf._output_shapes = 5 : i32})
    -> (tensor<*xi32>) {
  return(%arg0) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_not_shape_arr(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*xi32> {tfg.regenerate_output_shapes})
tfg.func @test_not_shape_arr(%arg0: tensor<*xi32> {tf._output_shapes = [5 : i32]})
    -> (tensor<*xi32>) {
  return(%arg0) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_wrong_shape_list_size(
// CHECK-SAME: %[[ARG0:.*]]: tensor<*xi32> {tfg.regenerate_output_shapes})
tfg.func @test_wrong_shape_list_size(%arg0: tensor<*xi32> {tf._output_shapes = [
  #tf_type.shape<2>, #tf_type.shape<2>
]}) -> (tensor<*xi32>) {
  return(%arg0) : tensor<*xi32>
}
