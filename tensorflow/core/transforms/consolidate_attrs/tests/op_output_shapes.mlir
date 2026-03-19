// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: A {tfg.regenerate_output_shapes} : () -> (tensor<4xi32>)
  %A, %ctl = A {_output_shapes = [#tf_type.shape<4>]} : () -> (tensor<*xi32>)
}

// -----

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: %[[A:.*]], %{{.*}} = A {tfg.regenerate_output_shapes} : () -> (tensor<4xi32>)
  %A, %ctl = A {_output_shapes = [#tf_type.shape<4>]} : () -> (tensor<*xi32>)
  // CHECK: Sink(%[[A]]) : tensor<4xi32>
  %ctl_0 = Sink(%A) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_result_type
// CHECK-NEXT: -> (tensor<4xi32>)
tfg.func @test_result_type(%arg0: tensor<i32>) -> (tensor<*xi32>) {
  // CHECK: %[[A:.*]], %{{.*}} = A {tfg.regenerate_output_shapes} : () -> (tensor<4xi32>)
  %A, %ctl = A {_output_shapes = [#tf_type.shape<4>]} : () -> (tensor<*xi32>)
  // CHECK: return(%[[A]]) : tensor<4xi32>
  return(%A) : tensor<*xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_ignore_invalid_shape
tfg.func @test_ignore_invalid_shape(%arg0: tensor<*xi32>) -> (tensor<*xi32>) {
  // CHECK: %[[A:.*]], %{{.*}} = A {_output_shapes = []} : () -> (tensor<*xi32>)
  %A, %ctl = A {_output_shapes = []} : () -> (tensor<*xi32>)
  return(%A) : tensor<*xi32>
}
