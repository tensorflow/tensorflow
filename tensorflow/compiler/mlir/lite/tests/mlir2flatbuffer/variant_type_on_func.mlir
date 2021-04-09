// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:  operator_codes: [  ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      type: VARIANT,
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0 ],
// CHECK-NEXT:    outputs: [ 0 ],
// CHECK-NEXT:    operators: [  ],
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 2
// CHECK-NEXT:  } ],
// CHECK-NEXT:  signature_defs: [  ]
// CHECK-NEXT: }
func @main(%arg0 : tensor<!tf.variant<tensor<2xi32>>>) -> tensor<!tf.variant<tensor<2xi32>>> {
  return %arg0 : tensor<!tf.variant<tensor<2xi32>>>
}
