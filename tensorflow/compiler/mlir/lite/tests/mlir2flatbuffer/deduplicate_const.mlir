// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s --check-prefix=CHECK
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer -disable-buffer-deduping %s -o - | flatbuffer_to_string - | FileCheck %s --check-prefix=NO_DEDUPE

module {
func.func @add(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {tf.entry_function = {inputs = "serving_default_x", outputs = "outputs"}} {
  %0 = "tfl.pseudo_const" () {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  %1 = "tfl.add" (%0, %arg0) {fused_activation_function = "NONE"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  func.return %1 : tensor<3x2xf32>
}

func.func @sub(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {tf.entry_function = {inputs = "serving_default_x", outputs = "outputs"}} {
  %0 = "tfl.pseudo_const" () {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  %1 = "tfl.sub" (%0, %arg0) {fused_activation_function = "NONE"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  func.return %1 : tensor<3x2xf32>
}
}

// CHECK:      {
// CHECK:        subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "serving_default_x",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "tfl.pseudo_const",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "outputs",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     } ],
// CHECK:          name: "add"
// CHECK-NEXT:   }, {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 4,
// CHECK-NEXT:       name: "serving_default_x",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "tfl.pseudo_const1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 6,
// CHECK-NEXT:       name: "outputs",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0 ],
// CHECK-NEXT:     outputs: [ 2 ],
// CHECK:          name: "sub"
// CHECK-NEXT:   } ],
// CHECK-NEXT:   description: "MLIR Converted.",
// CHECK-NEXT:   buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64 ]
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 49, 46, 54, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   } ],
// CHECK:      }

// NO_DEDUPE: {
// NO_DEDUPE:   version: 3,
// NO_DEDUPE:   operator_codes: [ {
// NO_DEDUPE:     version: 1
// NO_DEDUPE:   }, {
// NO_DEDUPE:     deprecated_builtin_code: 41,
// NO_DEDUPE:     version: 1,
// NO_DEDUPE:     builtin_code: SUB
// NO_DEDUPE:   } ],
// NO_DEDUPE:   subgraphs: [ {
// NO_DEDUPE:     tensors: [ {
// NO_DEDUPE:       shape: [ 3, 2 ],
// NO_DEDUPE:       buffer: 1,
// NO_DEDUPE:       name: "serving_default_x",
// NO_DEDUPE:       quantization: {
// NO_DEDUPE:       },
// NO_DEDUPE:       has_rank: true
// NO_DEDUPE:     }, {
// NO_DEDUPE:       shape: [ 3, 2 ],
// NO_DEDUPE:       buffer: 2,
// NO_DEDUPE:       name: "tfl.pseudo_const",
// NO_DEDUPE:       quantization: {
// NO_DEDUPE:       },
// NO_DEDUPE:       has_rank: true
// NO_DEDUPE:     }, {
// NO_DEDUPE:       shape: [ 3, 2 ],
// NO_DEDUPE:       buffer: 3,
// NO_DEDUPE:       name: "outputs",
// NO_DEDUPE:       quantization: {
// NO_DEDUPE:       },
// NO_DEDUPE:       has_rank: true
// NO_DEDUPE:     } ],
// NO_DEDUPE:     inputs: [ 0 ],
// NO_DEDUPE:     outputs: [ 2 ],
// NO_DEDUPE:     operators: [ {
// NO_DEDUPE:       inputs: [ 1, 0 ],
// NO_DEDUPE:       outputs: [ 2 ],
// NO_DEDUPE:       builtin_options_type: AddOptions,
// NO_DEDUPE:       builtin_options: {
// NO_DEDUPE:       }
// NO_DEDUPE:     } ],
// NO_DEDUPE:     name: "add"
// NO_DEDUPE:   }, {
// NO_DEDUPE:     tensors: [ {
// NO_DEDUPE:       shape: [ 3, 2 ],
// NO_DEDUPE:       buffer: 4,
// NO_DEDUPE:       name: "serving_default_x",
// NO_DEDUPE:       quantization: {
// NO_DEDUPE:       },
// NO_DEDUPE:       has_rank: true
// NO_DEDUPE:     }, {
// NO_DEDUPE:       shape: [ 3, 2 ],
// NO_DEDUPE:       buffer: 5,
// NO_DEDUPE:       name: "tfl.pseudo_const1",
// NO_DEDUPE:       quantization: {
// NO_DEDUPE:       },
// NO_DEDUPE:       has_rank: true
// NO_DEDUPE:     }, {
// NO_DEDUPE:       shape: [ 3, 2 ],
// NO_DEDUPE:       buffer: 6,
// NO_DEDUPE:       name: "outputs",
// NO_DEDUPE:       quantization: {
// NO_DEDUPE:       },
// NO_DEDUPE:       has_rank: true
// NO_DEDUPE:     } ],
// NO_DEDUPE:     inputs: [ 0 ],
// NO_DEDUPE:     outputs: [ 2 ],
// NO_DEDUPE:     operators: [ {
// NO_DEDUPE:       opcode_index: 1,
// NO_DEDUPE:       inputs: [ 1, 0 ],
// NO_DEDUPE:       outputs: [ 2 ],
// NO_DEDUPE:       builtin_options_type: SubOptions,
// NO_DEDUPE:       builtin_options: {
// NO_DEDUPE:       }
// NO_DEDUPE:     } ],
// NO_DEDUPE:     name: "sub"
// NO_DEDUPE:   } ],
// NO_DEDUPE:   description: "MLIR Converted.",
// NO_DEDUPE:   buffers: [ {
// NO_DEDUPE:   }, {
// NO_DEDUPE:   }, {
// NO_DEDUPE:     data: [ 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64 ]
// NO_DEDUPE:   }, {
// NO_DEDUPE:   }, {
// NO_DEDUPE:   }, {
// NO_DEDUPE:     data: [ 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64 ]
// NO_DEDUPE:   }, {
// NO_DEDUPE:   }, {
// NO_DEDUPE:     data: [ 49, 46, 54, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// NO_DEDUPE:   } ],
// NO_DEDUPE:   metadata: [ {
// NO_DEDUPE:     name: "min_runtime_version",
// NO_DEDUPE:     buffer: 7
// NO_DEDUPE:   } ],
// NO_DEDUPE:   signature_defs: [  ]
// NO_DEDUPE: }
