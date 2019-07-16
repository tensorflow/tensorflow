// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     builtin_code: LESS
// CHECK-NEXT:   }, {
// CHECK-NEXT:     builtin_code: CUSTOM,
// CHECK-NEXT:     custom_code: "Experimental_If"
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     builtin_code: MUL
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 1 ],
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "tfl.pseudo_input",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 1 ],
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "tfl.pseudo_input1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 1 ],
// CHECK-NEXT:       type: BOOL,
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "tfl.less",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 1 ],
// CHECK-NEXT:       buffer: 4,
// CHECK-NEXT:       name: "tf.If",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1 ],
// CHECK-NEXT:     outputs: [ 3 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1 ],
// CHECK-NEXT:       outputs: [ 2 ]
// CHECK-NEXT:     }, {
// CHECK-NEXT:       opcode_index: 1,
// CHECK-NEXT:       inputs: [ 2, 0, 1 ],
// CHECK-NEXT:       outputs: [ 3 ],
// CHECK-NEXT:       custom_options: [ 116, 104, 101, 110, 95, 115, 117, 98, 103, 114, 97, 112, 104, 95, 105, 110, 100, 101, 120, 0, 101, 108, 115, 101, 95, 115, 117, 98, 103, 114, 97, 112, 104, 95, 105, 110, 100, 101, 120, 0, 2, 21, 42, 2, 1, 2, 2, 1, 4, 4, 4, 36, 1 ]
// CHECK-NEXT:     } ]
// CHECK-NEXT:     name: "main"
// CHECK-NEXT:   }, {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       buffer: 5,
// CHECK-NEXT:       name: "arg0",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       buffer: 6,
// CHECK-NEXT:       name: "arg1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       buffer: 7,
// CHECK-NEXT:       name: "tfl.add",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1 ],
// CHECK-NEXT:     outputs: [ 2 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       opcode_index: 2,
// CHECK-NEXT:       inputs: [ 0, 1 ],
// CHECK-NEXT:       outputs: [ 2 ],
// CHECK-NEXT:       builtin_options_type: AddOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ]
// CHECK-NEXT:     name: "cond_true"
// CHECK-NEXT:   }, {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       buffer: 8,
// CHECK-NEXT:       name: "arg0",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       buffer: 9,
// CHECK-NEXT:       name: "arg1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       buffer: 10,
// CHECK-NEXT:       name: "tfl.mul",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1 ],
// CHECK-NEXT:     outputs: [ 2 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       opcode_index: 3,
// CHECK-NEXT:       inputs: [ 0, 1 ],
// CHECK-NEXT:       outputs: [ 2 ],
// CHECK-NEXT:       builtin_options_type: MulOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ]
// CHECK-NEXT:     name: "cond_false"
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
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   } ]
// CHECK-NEXT: }

func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tfl.pseudo_input"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "tfl.pseudo_input"(%arg1) : (tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.less"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %3 = "tf.If"(%2, %0, %1) {else_branch = @cond_false, then_branch = @cond_true} : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %3 : tensor<1xf32>
}

func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  return %0 : tensor<*xf32>
}
