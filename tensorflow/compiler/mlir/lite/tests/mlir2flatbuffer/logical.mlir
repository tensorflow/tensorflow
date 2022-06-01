// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func.func @main(tensor<4xi1>) -> tensor<4xi1> {
^bb0(%arg0: tensor<4xi1>):
  // CHECK:      {
  // CHECK-NEXT:   version: 3,
  // CHECK-NEXT:   operator_codes: [ {
  // CHECK-NEXT:     deprecated_builtin_code: 84,
  // CHECK-NEXT:     version: 1,
  // CHECK-NEXT:     builtin_code: LOGICAL_OR
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     deprecated_builtin_code: 86,
  // CHECK-NEXT:     version: 1,
  // CHECK-NEXT:     builtin_code: LOGICAL_AND
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   subgraphs: [ {
  // CHECK-NEXT:     tensors: [ {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 1,
  // CHECK-NEXT:       name: "arg0",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 2,
  // CHECK-NEXT:       name: "Const1",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 3,
  // CHECK-NEXT:       name: "Const2",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 4,
  // CHECK-NEXT:       name: "logical_or",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       type: BOOL,
  // CHECK-NEXT:       buffer: 5,
  // CHECK-NEXT:       name: "logical_and",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     } ],
  // CHECK-NEXT:     inputs: [ 0 ],
  // CHECK-NEXT:     outputs: [ 4 ],
  // CHECK-NEXT:     operators: [ {
  // CHECK-NEXT:       inputs: [ 0, 2 ],
  // CHECK-NEXT:       outputs: [ 3 ]
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       opcode_index: 1,
  // CHECK-NEXT:       inputs: [ 3, 1 ],
  // CHECK-NEXT:       outputs: [ 4 ]
  // CHECK-NEXT:     } ]
  // CHECK-NEXT:    name: "main"
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   description: "MLIR Converted.",
  // CHECK-NEXT:   buffers: [ {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     data: [ 1, 1, 1, 1 ]
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     data: [ 0, 0, 0, 0 ]
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     data: [ 49, 46, 49, 49, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   metadata: [ {
  // CHECK-NEXT:   name: "min_runtime_version",
  // CHECK-NEXT:   buffer: 6
  // CHECK-NEXT:   } ]
  // CHECK-NEXT:   signature_defs: [ ]
  // CHECK-NEXT: }
  // CHECK-EMPTY:

  %0 = "tfl.pseudo_const" () {value = dense<true> : tensor<4xi1>} : () -> tensor<4xi1> loc("Const1")
  %1 = "tfl.pseudo_const" () {value = dense<false> : tensor<4xi1>} : () -> tensor<4xi1> loc("Const2")
  %2 = "tfl.logical_or"(%arg0, %1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1> loc("logical_or")
  %3 = "tfl.logical_and"(%2, %0) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1> loc("logical_and")
  func.return %3 : tensor<4xi1>
}
