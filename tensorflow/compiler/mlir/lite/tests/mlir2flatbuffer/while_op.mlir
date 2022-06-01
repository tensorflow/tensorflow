// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     deprecated_builtin_code: 119,
// CHECK-NEXT:     version: 1,
// CHECK-NEXT:     builtin_code: WHILE
// CHECK-NEXT:   }, {
// CHECK-NEXT:     deprecated_builtin_code: 61,
// CHECK-NEXT:     version: 1,
// CHECK-NEXT:     builtin_code: GREATER
// CHECK-NEXT:   }, {
// CHECK-NEXT:     deprecated_builtin_code: 41,
// CHECK-NEXT:     version: 1,
// CHECK-NEXT:     builtin_code: SUB
// CHECK-NEXT:   }, {
// CHECK-NEXT:     version: 1
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "arg0",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 1 ],
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "arg1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "tfl.while",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 1 ],
// CHECK-NEXT:       buffer: 4,
// CHECK-NEXT:       name: "tfl.while:1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1 ],
// CHECK-NEXT:     outputs: [ 3 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1 ],
// CHECK-NEXT:       outputs: [ 2, 3 ],
// CHECK-NEXT:       builtin_options_type: WhileOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-NEXT:         cond_subgraph_index: 1,
// CHECK-NEXT:         body_subgraph_index: 2
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     name: "main"
// CHECK-NEXT:   }, {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       type: INT32,
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
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 7,
// CHECK-NEXT:       name: "Const",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       type: BOOL,
// CHECK-NEXT:       buffer: 8,
// CHECK-NEXT:       name: "tfl.greater",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1 ],
// CHECK-NEXT:     outputs: [ 3 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       opcode_index: 1,
// CHECK-NEXT:       inputs: [ 0, 2 ],
// CHECK-NEXT:       outputs: [ 3 ]
// CHECK-NEXT:     } ],
// CHECK-NEXT:     name: "cond"
// CHECK-NEXT:   }, {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 9,
// CHECK-NEXT:       name: "arg0",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       buffer: 10,
// CHECK-NEXT:       name: "arg1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 11,
// CHECK-NEXT:       name: "Const1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 12,
// CHECK-NEXT:       name: "tfl.sub",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [  ],
// CHECK-NEXT:       buffer: 13,
// CHECK-NEXT:       name: "tfl.add",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1 ],
// CHECK-NEXT:     outputs: [ 3, 4 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       opcode_index: 2,
// CHECK-NEXT:       inputs: [ 0, 2 ],
// CHECK-NEXT:       outputs: [ 3 ],
// CHECK-NEXT:       builtin_options_type: SubOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       opcode_index: 3,
// CHECK-NEXT:       inputs: [ 1, 1 ],
// CHECK-NEXT:       outputs: [ 4 ],
// CHECK-NEXT:       builtin_options_type: AddOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     name: "body"
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
// CHECK-NEXT:     data: [ 0, 0, 0, 0 ]
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 1, 0, 0, 0 ]
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 49, 46, 49, 53, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   } ],
// CHECK-NEXT:   metadata: [ {
// CHECK-NEXT:   name: "min_runtime_version",
// CHECK-NEXT:   buffer: 14
// CHECK-NEXT:   } ]
// CHECK-NEXT:  signature_defs: [ ]
// CHECK-NEXT: }

func.func @main(%arg0: tensor<i32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0:2 = "tfl.while"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
    %1 = func.call @cond(%arg2, %arg3) : (tensor<*xi32>, tensor<*xf32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
    %1:2 = func.call @body(%arg2, %arg3) : (tensor<*xi32>, tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>)
    "tfl.yield"(%1#0, %1#1) : (tensor<*xi32>, tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>)
  func.return %0#1 : tensor<1xf32>
}

func.func @cond(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> tensor<i1> {
  %cst = arith.constant dense<0> : tensor<i32> loc("Const")
  %0 = "tfl.greater"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @body(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>) {
  %cst = arith.constant dense<1> : tensor<i32> loc("Const")
  %0 = "tfl.sub"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %1 = tfl.add %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0, %1 : tensor<*xi32>, tensor<*xf32>
}
