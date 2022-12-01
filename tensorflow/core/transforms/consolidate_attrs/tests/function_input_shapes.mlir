// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_one
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xi32>
// CHECK-NEXT: -> (tensor<*xi32>)
// CHECK: attributes {tfg.regenerate_input_shapes}
tfg.func @test_one(%arg0: tensor<*xi32>) -> (tensor<*xi32>)
    attributes {tf._input_shapes = [#tf_type.shape<4>]} {
  // CHECK: A(%[[ARG0]]) : (tensor<4xi32>)
  %A, %ctl = A(%arg0) : (tensor<*xi32>) -> (tensor<*xi32>)
  return(%A) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_incompatible
// CHECK: %[[ARG0:.*]]: tensor<4xi32>
// CHECK: %[[ARG1:.*]]: tensor<i32>
// CHECK-NEXT: -> (tensor<*xi32>)
// CHECK: attributes {tfg.regenerate_input_shapes}
tfg.func @test_incompatible(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> (tensor<*xi32>)
    attributes {tf._input_shapes = [#tf_type.shape<4>, #tf_type.shape<4>]} {
  // CHECK: A(%[[ARG0]], %[[ARG1]]) : (tensor<4xi32>, tensor<i32>)
  %A, %ctl = A(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi32>)
  return(%A) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_return_type
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xi32>
// CHECK-NEXT: -> (tensor<4xi32>)
// CHECK: attributes {tfg.regenerate_input_shapes}
tfg.func @test_return_type(%arg0: tensor<*xi32>) -> (tensor<*xi32>)
    attributes {tf._input_shapes = [#tf_type.shape<4>]} {
  // CHECK: return(%[[ARG0]]) : tensor<4xi32>
  return(%arg0) : tensor<*xi32>
}

// -----

// Check _input_shapes is not ArrayAttr is left alone.

// CHECK-LABEL: tfg.func @test_not_array_attr
// CHECK: %[[ARG0:.*]]: tensor<*xi32>
// CHECK: attributes {tf._input_shapes = 5 : i32}
tfg.func @test_not_array_attr(%arg0: tensor<*xi32>) -> (tensor<*xi32>)
    attributes {tf._input_shapes = 5 : i32} {
  return(%arg0) : tensor<*xi32>
}

// -----

// Check _input_shapes is not an array of shapes is left alone.

// CHECK-LABEL: tfg.func @test_not_shape_arr
// CHECK: %[[ARG0:.*]]: tensor<*xi32>
// CHECK: attributes {tf._input_shapes = [5 : i32]}
tfg.func @test_not_shape_arr(%arg0: tensor<*xi32>) -> (tensor<*xi32>)
    attributes {tf._input_shapes = [5 : i32]} {
  return(%arg0) : tensor<*xi32>
}

// -----

// Check _input_shapes is an array of an incorrect number of shapes is left
// alone.

// CHECK-LABEL: tfg.func @test_wrong_shape_list_size
// CHECK: %[[ARG0:.*]]: tensor<*xi32>
// CHECK: attributes {tf._input_shapes = [#tf_type.shape<2>, #tf_type.shape<2>]}
tfg.func @test_wrong_shape_list_size(%arg0: tensor<*xi32>) -> (tensor<*xi32>)
    attributes {tf._input_shapes = [#tf_type.shape<2>, #tf_type.shape<2>]} {
  return(%arg0) : tensor<*xi32>
}

// -----

// Check that empty _input_shapes are ignored.

// CHECK-LABEL: tfg.func @test_empty_input_shapes
// CHECK: attributes {tf._input_shapes = []}
tfg.func @test_empty_input_shapes(%arg0: tensor<*xi32>) -> (tensor<*xi32>)
    attributes {tf._input_shapes = []} {
  return(%arg0) : tensor<*xi32>
}

// CHECK-LABEL: tfg.func @test_empty_input_shapes_unit
// CHECK: attributes {tf._input_shapes}
tfg.func @test_empty_input_shapes_unit(%arg0: tensor<*xi32>) -> (tensor<*xi32>)
    attributes {tf._input_shapes} {
  return(%arg0) : tensor<*xi32>
}
