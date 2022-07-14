// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// CHECK: {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:   operator_codes: [  ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       type: VARIANT,
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "tf.Const",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [  ],
// CHECK-NEXT:     outputs: [ 0 ],
// CHECK-NEXT:     operators: [  ],
// CHECK-NEXT:     name: "main"
// CHECK-NEXT:   } ],
// CHECK-NEXT:   description: "MLIR Converted.",
// CHECK-NEXT:   buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 128, 0, 0, 0, 128, 0, 0, 0 ]
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   } ],
// CHECK-NEXT:   metadata: [ {
// CHECK-NEXT:     name: "min_runtime_version",
// CHECK-NEXT:     buffer: 2
// CHECK-NEXT:   } ],
// CHECK-NEXT:   signature_defs: [  ]
// CHECK-NEXT: }
func.func @main() -> tensor<!tf_type.variant<tensor<2xi32>>> {
  %0 = "tf.Const"() {device = "", name = "", dtype = "tfdtype$DT_INT32", value = opaque<"tf", "0x746674656E736F722464747970653A2044545F494E5433320A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20320A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3230305C3030305C3030305C3030305C3230305C3030305C3030305C303030220A"> : tensor<!tf_type.variant>} : () -> tensor<!tf_type.variant<tensor<2xi32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<2xi32>>>
}
