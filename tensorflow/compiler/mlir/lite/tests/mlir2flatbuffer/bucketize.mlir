// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<3x2xf32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<3x2xf32>):
  // CHECK:      {
  // CHECK-NEXT:     version: 3,
  // CHECK-NEXT:     operator_codes: [ {
  // CHECK-NEXT:       deprecated_builtin_code: 127,
  // CHECK-NEXT:       version: 1,
  // CHECK-NEXT:       builtin_code: BUCKETIZE
  // CHECK-NEXT:     } ],
  // CHECK-NEXT:     subgraphs: [ {
  // CHECK-NEXT:       tensors: [ {
  // CHECK-NEXT:         shape: [ 3, 2 ],
  // CHECK-NEXT:         buffer: 1,
  // CHECK-NEXT:         name: "arg0",
  // CHECK-NEXT:         quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:         }
  // CHECK-NEXT:       }, {
  // CHECK-NEXT:         shape: [ 3, 2 ],
  // CHECK-NEXT:         buffer: 2,
  // CHECK-NEXT:         name: "Const",
  // CHECK-NEXT:         quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:         }
  // CHECK-NEXT:       }, {
  // CHECK-NEXT:         shape: [ 3, 2 ],
  // CHECK-NEXT:         type: INT32,
  // CHECK-NEXT:         buffer: 3,
  // CHECK-NEXT:         name: "bucketize",
  // CHECK-NEXT:         quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:         }
  // CHECK-NEXT:       } ],
  // CHECK-NEXT:       inputs: [ 0 ],
  // CHECK-NEXT:       outputs: [ 2 ],
  // CHECK-NEXT:       operators: [ {
  // CHECK-NEXT:         inputs: [ 1 ],
  // CHECK-NEXT:         outputs: [ 2 ],
  // CHECK-NEXT:         builtin_options_type: BucketizeOptions,
  // CHECK-NEXT:         builtin_options: {
  // CHECK-NEXT:           boundaries: [ 0.0, 10.0, 100.0 ]
  // CHECK-NEXT:         }
  // CHECK-NEXT:       } ],
  // CHECK-NEXT:       name: "main"
  // CHECK-NEXT:     } ],
  // CHECK-NEXT:     description: "MLIR Converted.",
  // CHECK-NEXT:     buffers: [ {
  // CHECK-EMPTY:
  // CHECK-NEXT:     }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       data: [ 0, 0, 160, 192, 0, 64, 28, 70, 0, 0, 22, 67, 0, 0, 32, 65, 0, 0, 160, 64, 0, 0, 200, 66 ]
  // CHECK-NEXT:     }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       data: [ 50, 46, 56, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
  // CHECK-NEXT:     } ],
  // CHECK-NEXT:     metadata: [ {
  // CHECK-NEXT:       name: "min_runtime_version",
  // CHECK-NEXT:       buffer: 4
  // CHECK-NEXT:     } ],
  // CHECK-NEXT:     signature_defs: [  ]
  // CHECK-NEXT: }
  // CHECK-EMPTY:

  %0 = "tfl.pseudo_const" () {value = dense<[[-5.0, 10000.0], [150.0, 10.0], [5.0, 100.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32> loc("Const")
  %1 = "tfl.bucketize"(%0) {boundaries = [0.0 : f32, 10.0 : f32, 100.0 : f32]} : (tensor<3x2xf32>) -> tensor<3x2xi32> loc("bucketize")
  return %1 : tensor<3x2xi32>
}
