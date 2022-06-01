// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func.func @main(tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32> {
^bb0(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>):
// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     deprecated_builtin_code: 127,
// CHECK-NEXT:     version: 1,
// CHECK-NEXT:     builtin_code: UNSORTED_SEGMENT_PROD
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 8 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "arg0",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 8 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "arg1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 8 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "tfl.unsorted_segment_prod",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1 ],
// CHECK-NEXT:     outputs: [ 2 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1 ],
// CHECK-NEXT:       outputs: [ 2 ]
// CHECK-NEXT:       builtin_options_type: UnsortedSegmentProdOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-NEXT:       num_segments: 8
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
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 50, 46, 49, 48, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   } ],
// CHECK-NEXT:   metadata: [ {
// CHECK-NEXT:   name: "min_runtime_version",
// CHECK-NEXT:   buffer: 4
// CHECK-NEXT:   } ]
// CHECK-NEXT:   signature_defs: [ ]
// CHECK-NEXT: }
  %0 = "tfl.unsorted_segment_prod"(%arg0, %arg1) {num_segments = 8 : i32} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>
}
