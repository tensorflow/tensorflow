// RUN: tfg-transforms-opt %s --tfg-drop-unregistered-output-shapes=skip=tfg.placeholder | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: placeholder
  // CHECK: _output_shapes
  // CHECK: AddV2
  // CHECK-NOT: _output_shapes
  // CHECK: placeholder
  %arg0, %ctl = "tfg.placeholder"() {_output_shapes = ["tfshape$dim {size = 1}"] } : () -> (tensor<*xi32>, !tf_type.control)
  %add, %ctl3 = "tfg.AddV2"(%arg0, %arg1) {_output_shapes = ["tfshape$dim {size = 1}"] } : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, !tf_type.control)
  %arg1, %ctl2 = "tfg.placeholder"()  : () -> (tensor<*xi32>, !tf_type.control)
}

