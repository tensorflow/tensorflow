// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-custom-ops -emit-builtin-tflite-ops=false -o - | flatbuffer_to_string - | FileCheck %s

func.func @main() -> tensor<4xi4> {
  // CHECK: {
  // CHECK:   version: 3,
  // CHECK:   operator_codes: [  ],
  // CHECK:   subgraphs: [ {
  // CHECK:     tensors: [ {
  // CHECK:       shape: [ 4 ],
  // CHECK:       type: INT4,
  // CHECK:       buffer: 1,
  // CHECK:       name: "Const",
  // CHECK:       quantization: {
  // CHECK-EMPTY
  // CHECK:       },
  // CHECK:       has_rank: true
  // CHECK:     } ],
  // CHECK:     inputs: [  ],
  // CHECK:     outputs: [ 0 ],
  // CHECK:     operators: [  ],
  // CHECK:     name: "main"
  // CHECK:   } ],
  // CHECK:   description: "MLIR Converted.",
  // CHECK:   buffers: [ {
  // CHECK-EMPTY
  // CHECK:   }, {
  // CHECK:     data: [ 56, 190 ]
  // CHECK:   }, {
  // CHECK:     data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
  // CHECK:   } ],
  // CHECK:   metadata: [ {
  // CHECK:     name: "min_runtime_version",
  // CHECK:     buffer: 2
  // CHECK:   } ],
  // CHECK:   signature_defs: [  ]
  // CHECK: }

  // Test that i4 buffers are densely packed, i.e. [-8, 3, -2, -5] should be
  // be packed low-bits-first as [0x38, 0xBE] or [56, 190]. Tensor type should
  // be INT4.
  %0 = "tfl.pseudo_const" () {value = dense<[-8, 3, -2, -5]> : tensor<4xi4>} : () -> tensor<4xi4> loc("Const")
  func.return %0 : tensor<4xi4>
}
