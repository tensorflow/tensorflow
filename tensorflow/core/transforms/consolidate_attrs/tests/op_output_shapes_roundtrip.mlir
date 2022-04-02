// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --tfg-prepare-attrs-export | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: A {_output_shapes = [#tf_type.shape<4>]} : () -> (tensor<4xi32>)
  %A, %ctl = A {_output_shapes = [#tf_type.shape<4>]} : () -> (tensor<*xi32>)
}
