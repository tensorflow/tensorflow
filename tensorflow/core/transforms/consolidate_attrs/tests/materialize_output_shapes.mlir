// RUN: tfg-transforms-opt --tfg-prepare-attrs-export %s | FileCheck %s

tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: A {_output_shapes = [#tf_type.shape<4>, #tf_type.shape<8>]}
  %A:2, %ctlA = A {tfg.regenerate_output_shapes} : () -> (tensor<4xi32>, tensor<8xi32>)
  // CHECK: B {_output_shapes = [#tf_type.shape<*>, #tf_type.shape<2>]}
  %B:2, %ctlB = B {tfg.regenerate_output_shapes} : () -> (tensor<*xi32>, tensor<2xi32>)
  // Test excludes ops without regenerate attribute.
  // CHECK: C : ()
  %C:2, %ctlC = C : () -> (tensor<*xi32>, tensor<4xi32>)
}
