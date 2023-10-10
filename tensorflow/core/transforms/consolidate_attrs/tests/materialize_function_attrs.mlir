// RUN: tfg-transforms-opt --tfg-prepare-attrs-export %s | FileCheck %s

// CHECK-LABEL: tfg.func @test_func_attrs
// CHECK-SAME: %[[ARG0:.*]]: tensor<{{.*}}> {tf._output_shapes = [#tf_type.shape<4>]}
// CHECK-NEXT: %[[ARG1:.*]]: tensor<{{.*}}> {tf._output_shapes = [#tf_type.shape<2>]
// CHECK-SAME:   tfg.handle_data = [tensor<8xi32>]}
// CHECK-NEXT: -> (tensor<{{.*}}> {tf._output_shapes = [#tf_type.shape<10>]}
// CHECK-NEXT:     tensor<{{.*}}> {tf._output_shapes = [#tf_type.shape<20>],
// CHECK-SAME:   tfg.handle_data = [tensor<4xi32>, tensor<8xf32>]}
// CHECK-NEXT: attributes {tf._input_shapes = [#tf_type.shape<4>, #tf_type.shape<2>]}
tfg.func @test_func_attrs(%arg0: tensor<4xi32> {tfg.regenerate_output_shapes},
                          %arg1: tensor<2x!tf_type.resource<tensor<8xi32>>> {tfg.regenerate_output_shapes})
    -> (tensor<10xi32> {tfg.regenerate_output_shapes},
        tensor<20x!tf_type.resource<tensor<4xi32>, tensor<8xf32>>> {tfg.regenerate_output_shapes})
    attributes {tfg.regenerate_input_shapes} {
  %A:2, %ctl = A : () -> (tensor<10xi32>, tensor<20x!tf_type.resource<tensor<4xi32>, tensor<8xf32>>>)
  return(%A#0, %A#1) : tensor<10xi32>, tensor<20x!tf_type.resource<tensor<4xi32>, tensor<8xf32>>>
}

// CHECK-LABEL: tfg.func @test_ignore_no_regenerate(
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xi32>)
// CHECK-NEXT: -> (tensor<4xi32>)
// CHECK-NOT: attributes
tfg.func @test_ignore_no_regenerate(%arg0: tensor<4xi32>) -> (tensor<4xi32>) {
  return(%arg0) : tensor<4xi32>
}

// CHECK-LABEL: tfg.func @test_is_ref
// CHECK-SAME: tfg.is_ref
tfg.func @test_is_ref(%arg0: tensor<*x!tf_type.int32ref>) -> (tensor<*xi32>) {
  %A, %ctl = A() : () -> (tensor<*xi32>)
  return(%A) : tensor<*xi32>
}
