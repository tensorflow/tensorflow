// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --tfg-prepare-attrs-export | FileCheck %s

// CHECK-LABEL: tfg.func generic @generic_func
// CHECK-SAME: !tf_type.tensor {tf._output_shapes = [#tf_type.shape<4>]}
tfg.func generic @generic_func(%arg0: !tf_type.tensor {tf._output_shapes = [#tf_type.shape<4>]})
  // CHECK-NEXT: !tf_type.tensor {tf._output_shapes = [#tf_type.shape<4>]}
  -> (!tf_type.tensor {tf._output_shapes = [#tf_type.shape<4>]})
    // CHECK-NEXT: attributes {tf._input_shapes = [#tf_type.shape<4>]}
    attributes {tf._input_shapes = [#tf_type.shape<4>]} {
  // CHECK-NEXT: A {_output_shapes = [#tf_type.shape<4>]}
  %A, %ctlA = A {_output_shapes = [#tf_type.shape<4>]} : () -> (!tf_type.tensor)
  return(%A) : !tf_type.tensor
}
