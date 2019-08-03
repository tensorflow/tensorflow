// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<3x2xi32>) -> tensor<6xi32> {
^bb0(%arg0: tensor<3x2xi32>):
// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     builtin_code: RESHAPE
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "Input",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 6 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "tfl.reshape",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0 ],
// CHECK-NEXT:     outputs: [ 1 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       inputs: [ 0 ],
// CHECK-NEXT:       outputs: [ 1 ],
// CHECK-NEXT:       builtin_options_type: ReshapeOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-NEXT:         new_shape: [ 6 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     } ]
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:   } ],
// CHECK-NEXT:   description: "MLIR Converted.",
// CHECK-NEXT:   buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   } ]
// CHECK-NEXT: }

  %0 = "tfl.pseudo_input" (%arg0) : (tensor<3x2xi32>) -> tensor<3x2xi32> loc("Input")
  %2 = "tfl.reshape" (%0) : (tensor<3x2xi32>) -> tensor<6xi32>
  return %2 : tensor<6xi32>
}
