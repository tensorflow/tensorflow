// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func.func @main(tensor<4x4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32> {
// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     deprecated_builtin_code: 44,
// CHECK-NEXT:     version: 1,
// CHECK-NEXT:     builtin_code: UNIDIRECTIONAL_SEQUENCE_LSTM
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 4, 4, 4 ],
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "arg0",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "arg1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "arg2",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 4,
// CHECK-NEXT:       name: "arg3",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 5,
// CHECK-NEXT:       name: "arg4",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 6,
// CHECK-NEXT:       name: "arg5",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 7,
// CHECK-NEXT:       name: "arg6",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 8,
// CHECK-NEXT:       name: "arg7",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 9,
// CHECK-NEXT:       name: "arg8",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 10,
// CHECK-NEXT:       name: "arg9",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 11,
// CHECK-NEXT:       name: "arg10",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 12,
// CHECK-NEXT:       name: "arg11",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 13,
// CHECK-NEXT:       name: "arg12",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 14,
// CHECK-NEXT:       name: "arg13",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 15,
// CHECK-NEXT:       name: "arg14",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 16,
// CHECK-NEXT:       name: "arg15",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 17,
// CHECK-NEXT:       name: "arg16",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4 ],
// CHECK-NEXT:       buffer: 18,
// CHECK-NEXT:       name: "arg17",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 19,
// CHECK-NEXT:       name: "arg18",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 20,
// CHECK-NEXT:       name: "arg19",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 21,
// CHECK-NEXT:       name: "arg20",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       buffer: 22,
// CHECK-NEXT:       name: "arg21",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       name: "Const",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       is_variable: true,
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4 ],
// CHECK-NEXT:       name: "Const1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       is_variable: true,
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       name: "input_to_input_intermediate",
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       name: "input_to_forget_intermediate",
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       name: "input_to_cell_intermediate",
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       name: "input_to_output_intermediate",
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       name: "effective_hidden_scale_intermediate",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.007788 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 4, 4, 4 ],
// CHECK-NEXT:       buffer: 25,
// CHECK-NEXT:       name: "tfl.unidirectional_sequence_lstm",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 ],
// CHECK-NEXT:     outputs: [ 29 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 18, 19, 20, 21 ],
// CHECK-NEXT:       outputs: [ 29 ],
// CHECK-NEXT:         builtin_options_type: UnidirectionalSequenceLSTMOptions,
// CHECK-NEXT:         builtin_options: {
// CHECK-NEXT:           time_major: true
// CHECK-NEXT:         },
// CHECK-NEXT:       intermediates: [ 24, 25, 26, 27, 28 ]
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
// CHECK-NEXT:     data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 49, 46, 49, 51, 46, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   } ],
// CHECK-NEXT:   metadata: [ {
// CHECK-NEXT:   name: "min_runtime_version",
// CHECK-NEXT:   buffer: 26
// CHECK-NEXT:   } ]
// CHECK-NEXT:   signature_defs: [ ]
// CHECK-NEXT: }
// CHECK-EMPTY:

^bb0(%arg0: tensor<4x4x4xf32>,
  %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>,
  %arg5: tensor<4x4xf32>, %arg6: tensor<4x4xf32>, %arg7: tensor<4x4xf32>, %arg8: tensor<4x4xf32>,
  %arg9: tensor<4xf32>, %arg10: tensor<4xf32>, %arg11: tensor<4xf32>,
  %arg12: tensor<4xf32>, %arg13: tensor<4xf32>, %arg14: tensor<4xf32>, %arg15: tensor<4xf32>,
  %arg16: tensor<4x4xf32>, %arg17: tensor<4xf32>,
  %arg18: tensor<4x4xf32>, %arg19: tensor<4x4xf32>, %arg20: tensor<4x4xf32>, %arg21: tensor<4x4xf32>):
  %0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32> loc("Const")
  %1 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32> loc("Const")
  %2 = "tfl.unidirectional_sequence_lstm"(%arg0,
    %arg1, %arg2, %arg3, %arg4,
    %arg5, %arg6, %arg7, %arg8,
    %arg9, %arg10, %arg11,
    %arg12, %arg13, %arg14, %arg15,
    %arg16, %arg17,
    %0, %1,
    %arg18, %arg19,%arg20, %arg21) {
      effective_hidden_scale_intermediate = tensor<0x!quant.uniform<i8<-127:127>:f32, 0.0077881771139800549>>,
      fused_activation_function = "NONE",
      input_to_cell_intermediate = tensor<0xf32>,
      input_to_forget_intermediate = tensor<0xf32>,
      input_to_input_intermediate = tensor<0xf32>,
      input_to_output_intermediate = tensor<0xf32>, time_major = true}
  : (tensor<4x4x4xf32>,
    tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>,
    tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>,
    tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
    tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
    tensor<4x4xf32>, tensor<4xf32>,
    tensor<4x4xf32>, tensor<4x4xf32>,
    tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>
  func.return %2 : tensor<4x4x4xf32>
}
