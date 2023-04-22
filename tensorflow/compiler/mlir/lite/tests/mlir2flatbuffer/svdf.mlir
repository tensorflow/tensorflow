// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>) -> tensor<4 x f32> {
// CHECK:      {
// CHECK-NEXT:     version: 3,
// CHECK-NEXT:     operator_codes: [ {
// CHECK-NEXT:       deprecated_builtin_code: 27,
// CHECK-NEXT:       version: 1,
// CHECK-NEXT:       builtin_code: SVDF
// CHECK-NEXT:     } ],
// CHECK-NEXT:     subgraphs: [ {
// CHECK-NEXT:       tensors: [ {
// CHECK-NEXT:         shape: [ 4 ],
// CHECK-NEXT:         buffer: 1,
// CHECK-NEXT:         name: "arg0",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         }
// CHECK-NEXT:       }, {
// CHECK-NEXT:         shape: [ 4 ],
// CHECK-NEXT:         buffer: 2,
// CHECK-NEXT:         name: "arg1",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         }
// CHECK-NEXT:       }, {
// CHECK-NEXT:         shape: [ 4 ],
// CHECK-NEXT:         buffer: 3,
// CHECK-NEXT:         name: "arg2",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         }
// CHECK-NEXT:       }, {
// CHECK-NEXT:         shape: [ 4 ],
// CHECK-NEXT:         buffer: 4,
// CHECK-NEXT:         name: "arg3",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         }
// CHECK-NEXT:       }, {
// CHECK-NEXT:         shape: [ 4 ],
// CHECK-NEXT:         name: "Const",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         },
// CHECK-NEXT:         is_variable: true
// CHECK-NEXT:       }, {
// CHECK-NEXT:         shape: [ 4 ],
// CHECK-NEXT:         buffer: 6,
// CHECK-NEXT:         name: "tfl.svdf",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         }
// CHECK-NEXT:       } ],
// CHECK-NEXT:       inputs: [ 0, 1, 2, 3 ],
// CHECK-NEXT:       outputs: [ 5 ],
// CHECK-NEXT:       operators: [ {
// CHECK-NEXT:         inputs: [ 0, 1, 2, 3, 4 ],
// CHECK-NEXT:         outputs: [ 5 ],
// CHECK-NEXT:         builtin_options_type: SVDFOptions,
// CHECK-NEXT:         builtin_options: {
// CHECK-NEXT:           rank: 2,
// CHECK-NEXT:           fused_activation_function: RELU
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
// CHECK-EMPTY:
// CHECK-NEXT:     }, {
// CHECK-EMPTY:
// CHECK-NEXT:     }, {
// CHECK-EMPTY:
// CHECK-NEXT:     }, {
// CHECK-NEXT:      data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:     }, {
// CHECK-EMPTY:
// CHECK-NEXT:     }, {
// CHECK-NEXT:      data: [ 49, 46, 53, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:     } ],
// CHECK-NEXT:     metadata: [ {
// CHECK-NEXT:     name: "min_runtime_version",
// CHECK-NEXT:     buffer: 7
// CHECK-NEXT:     } ]
// CHECK-NEXT:     signature_defs: [ ]
// CHECK-NEXT:   }
// CHECK-EMPTY:

^bb0(%arg0: tensor<4 x f32>, %arg1: tensor<4 x f32>, %arg2: tensor<4 x f32>, %arg3: tensor<4 x f32>):
  %0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %1 = "tfl.svdf"(%arg0, %arg1, %arg2, %arg3, %0) {fused_activation_function = "RELU", rank = 2 : i32} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
