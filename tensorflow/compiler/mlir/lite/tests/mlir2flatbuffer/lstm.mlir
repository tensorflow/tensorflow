// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>) -> tensor<4 x f32> {
// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     builtin_code: LSTM,
// CHECK-NEXT:     version: 1
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "arg0",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "arg1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "arg2",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 4,
// CHECK-NEXT:       name: "arg3",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 5,
// CHECK-NEXT:       name: "arg4",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 6,
// CHECK-NEXT:       name: "arg5",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 7,
// CHECK-NEXT:       name: "arg6",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 8,
// CHECK-NEXT:       name: "arg7",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 9,
// CHECK-NEXT:       name: "arg8",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 10,
// CHECK-NEXT:       name: "arg9",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 11,
// CHECK-NEXT:       name: "arg10",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 12,
// CHECK-NEXT:       name: "arg11",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 13,
// CHECK-NEXT:       name: "arg12",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 14,
// CHECK-NEXT:       name: "arg13",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 15,
// CHECK-NEXT:       name: "arg14",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 16,
// CHECK-NEXT:       name: "arg15",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 17,
// CHECK-NEXT:       name: "arg16",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 18,
// CHECK-NEXT:       name: "arg17",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 19,
// CHECK-NEXT:       name: "arg18",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 20,
// CHECK-NEXT:       name: "arg19",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 21,
// CHECK-NEXT:       name: "arg20",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 22,
// CHECK-NEXT:       name: "arg21",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       name: "Const",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       is_variable: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       name: "Const1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       is_variable: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 25,
// CHECK-NEXT:       name: "tfl.lstm",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ],
// CHECK-NEXT:     outputs: [ 24 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 18, 19, 20, 21 ],
// CHECK-NEXT:       outputs: [ 24 ],
// CHECK-NEXT:       builtin_options_type: LSTMOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     name: "main"
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
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   } ]
// CHECK-NEXT: }
// CHECK-EMPTY:


^bb0(%arg0: tensor<4 x f32>, %arg1: tensor<4 x f32>, %arg2: tensor<4 x f32>, %arg3: tensor<4 x f32>, %arg4: tensor<4 x f32>, %arg5: tensor<4 x f32>, %arg6: tensor<4 x f32>, %arg7: tensor<4 x f32>, %arg8: tensor<4 x f32>, %arg9: tensor<4 x f32>, %arg10: tensor<4 x f32>, %arg11: tensor<4 x f32>, %arg12: tensor<4 x f32>, %arg13: tensor<4 x f32>, %arg14: tensor<4 x f32>, %arg15: tensor<4 x f32>, %arg16: tensor<4 x f32>, %arg17: tensor<4 x f32>, %arg18: tensor<4 x f32>, %arg19: tensor<4 x f32>, %arg20: tensor<4 x f32>, %arg21: tensor<4 x f32>):
  %cst0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %cst1 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %24 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %cst0, %cst1, %arg18, %arg19, %arg20, %arg21) ({}) {fused_activation_function = "NONE", kernel_type = "FULL"} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %24 : tensor<4xf32>
}
