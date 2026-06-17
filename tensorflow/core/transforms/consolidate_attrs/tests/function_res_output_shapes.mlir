// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_output_shapes
// CHECK: -> (tensor<4xi32> {tfg.regenerate_output_shapes})
tfg.func @test_output_shapes(%arg0: tensor<*xi32>)
    -> (tensor<*xi32> {tf._output_shapes = [#tf_type.shape<4>]}) {
  // CHECK: %[[A:.*]], %{{.*}} = A(%{{.*}}) : (tensor<*xi32>) -> (tensor<4xi32>)
  %A, %ctl = A(%arg0) : (tensor<*xi32>) -> (tensor<*xi32>)
  // CHECK: return(%[[A]]) : tensor<4xi32>
  return(%A) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_output_shapes_arg_type(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xi32>)
// CHECK-NEXT: -> (tensor<4xi32> {tfg.regenerate_output_shapes})
tfg.func @test_output_shapes_arg_type(%arg0: tensor<*xi32>)
    -> (tensor<*xi32> {tf._output_shapes = [#tf_type.shape<4>]}) {
  // CHECK: return(%[[ARG0]]) : tensor<4xi32>
  return(%arg0) : tensor<*xi32>
}
