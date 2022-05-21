// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: A {tfg.regenerate_output_shapes}
  // CHECK-SAME: tensor<?xi32>, tensor<4xi32>, tensor<?xi32>
  %A:3, %ctl = A {
    _output_shapes = [#tf_type.shape<-4>, #tf_type.shape<4>, #tf_type.shape<?>]
  } : () -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>)
}
