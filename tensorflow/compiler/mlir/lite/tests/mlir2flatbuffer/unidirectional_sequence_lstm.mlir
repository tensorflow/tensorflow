// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>, tensor<4 x f32>) -> tensor<4 x f32> {
// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     builtin_code: UNIDIRECTIONAL_SEQUENCE_LSTM,
// CHECK-NEXT:     version: 1
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "tfl.pseudo_input",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "tfl.pseudo_input1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "tfl.pseudo_input2",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 4,
// CHECK-NEXT:       name: "tfl.pseudo_input3",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 5,
// CHECK-NEXT:       name: "tfl.pseudo_input4",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 6,
// CHECK-NEXT:       name: "tfl.pseudo_input5",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 7,
// CHECK-NEXT:       name: "tfl.pseudo_input6",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 8,
// CHECK-NEXT:       name: "tfl.pseudo_input7",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 9,
// CHECK-NEXT:       name: "tfl.pseudo_input8",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 10,
// CHECK-NEXT:       name: "tfl.pseudo_input9",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 11,
// CHECK-NEXT:       name: "tfl.pseudo_input10",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 12,
// CHECK-NEXT:       name: "tfl.pseudo_input11",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 13,
// CHECK-NEXT:       name: "tfl.pseudo_input12",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 14,
// CHECK-NEXT:       name: "tfl.pseudo_input13",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 15,
// CHECK-NEXT:       name: "tfl.pseudo_input14",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 16,
// CHECK-NEXT:       name: "tfl.pseudo_input15",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 17,
// CHECK-NEXT:       name: "tfl.pseudo_input16",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 18,
// CHECK-NEXT:       name: "tfl.pseudo_input17",
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
// CHECK-NEXT:       buffer: 21,
// CHECK-NEXT:       name: "tfl.pseudo_input18",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 22,
// CHECK-NEXT:       name: "tfl.pseudo_input19",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 23,
// CHECK-NEXT:       name: "tfl.pseudo_input20",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 24,
// CHECK-NEXT:       name: "tfl.pseudo_input21",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 25,
// CHECK-NEXT:       name: "tfl.unidirectional_sequence_lstm",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23 ],
// CHECK-NEXT:     outputs: [ 24 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 ],
// CHECK-NEXT:       outputs: [ 24 ],
// CHECK-NEXT:         builtin_options_type: UnidirectionalSequenceLSTMOptions,
// CHECK-NEXT:         builtin_options: {
// CHECK-NEXT:           time_major: true
// CHECK-NEXT:         }
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
// CHECK-NEXT:     data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
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
// CHECK-EMPTY:

^bb0(%arg0: tensor<4 x f32>, %arg1: tensor<4 x f32>, %arg2: tensor<4 x f32>, %arg3: tensor<4 x f32>, %arg4: tensor<4 x f32>, %arg5: tensor<4 x f32>, %arg6: tensor<4 x f32>, %arg7: tensor<4 x f32>, %arg8: tensor<4 x f32>, %arg9: tensor<4 x f32>, %arg10: tensor<4 x f32>, %arg11: tensor<4 x f32>, %arg12: tensor<4 x f32>, %arg13: tensor<4 x f32>, %arg14: tensor<4 x f32>, %arg15: tensor<4 x f32>, %arg16: tensor<4 x f32>, %arg17: tensor<4 x f32>, %arg20: tensor<4 x f32>, %arg21: tensor<4 x f32>, %arg22: tensor<4 x f32>, %arg23: tensor<4 x f32>):
  %0 = "tfl.pseudo_input" (%arg0) : (tensor<4 x f32>) -> tensor<4 x f32>
  %1 = "tfl.pseudo_input" (%arg1) : (tensor<4 x f32>) -> tensor<4 x f32>
  %2 = "tfl.pseudo_input" (%arg2) : (tensor<4 x f32>) -> tensor<4 x f32>
  %3 = "tfl.pseudo_input" (%arg3) : (tensor<4 x f32>) -> tensor<4 x f32>
  %4 = "tfl.pseudo_input" (%arg4) : (tensor<4 x f32>) -> tensor<4 x f32>
  %5 = "tfl.pseudo_input" (%arg5) : (tensor<4 x f32>) -> tensor<4 x f32>
  %6 = "tfl.pseudo_input" (%arg6) : (tensor<4 x f32>) -> tensor<4 x f32>
  %7 = "tfl.pseudo_input" (%arg7) : (tensor<4 x f32>) -> tensor<4 x f32>
  %8 = "tfl.pseudo_input" (%arg8) : (tensor<4 x f32>) -> tensor<4 x f32>
  %9 = "tfl.pseudo_input" (%arg9) : (tensor<4 x f32>) -> tensor<4 x f32>
  %10 = "tfl.pseudo_input" (%arg10) : (tensor<4 x f32>) -> tensor<4 x f32>
  %11 = "tfl.pseudo_input" (%arg11) : (tensor<4 x f32>) -> tensor<4 x f32>
  %12 = "tfl.pseudo_input" (%arg12) : (tensor<4 x f32>) -> tensor<4 x f32>
  %13 = "tfl.pseudo_input" (%arg13) : (tensor<4 x f32>) -> tensor<4 x f32>
  %14 = "tfl.pseudo_input" (%arg14) : (tensor<4 x f32>) -> tensor<4 x f32>
  %15 = "tfl.pseudo_input" (%arg15) : (tensor<4 x f32>) -> tensor<4 x f32>
  %16 = "tfl.pseudo_input" (%arg16) : (tensor<4 x f32>) -> tensor<4 x f32>
  %17 = "tfl.pseudo_input" (%arg17) : (tensor<4 x f32>) -> tensor<4 x f32>
  %18 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %19 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %20 = "tfl.pseudo_input" (%arg20) : (tensor<4 x f32>) -> tensor<4 x f32>
  %21 = "tfl.pseudo_input" (%arg21) : (tensor<4 x f32>) -> tensor<4 x f32>
  %22 = "tfl.pseudo_input" (%arg22) : (tensor<4 x f32>) -> tensor<4 x f32>
  %23 = "tfl.pseudo_input" (%arg23) : (tensor<4 x f32>) -> tensor<4 x f32>
  %24 = "tfl.unidirectional_sequence_lstm"(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23) {fused_activation_function = "NONE", time_major = true} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %24 : tensor<4xf32>
}
