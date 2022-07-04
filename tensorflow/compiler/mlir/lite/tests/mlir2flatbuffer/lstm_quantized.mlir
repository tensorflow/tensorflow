// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func.func @main(%arg0: tensor<1x528x!quant.uniform<i8:f32, 0.037248000502586365:-19>>, %arg1: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg2: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.031925998628139496>>, %arg3: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.056272000074386597>>, %arg4: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.063763998448848724>>, %arg5: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.013358999975025654>>, %arg6: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.022830000147223473>>, %arg7: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.032276000827550888>>, %arg8: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.035427000373601913>>, %arg9: tensor<2048x!quant.uniform<i32:f32, 4.2675782196965883E-7>>, %arg10: tensor<2048x!quant.uniform<i32:f32, 1.0742187583900886E-7>>, %arg11: tensor<2048x!quant.uniform<i32:f32, 1.6406249869760359E-7>>, %arg12: tensor<2048x!quant.uniform<i32:f32, 1.523437447303877E-7>>, %arg13: tensor<640x2048x!quant.uniform<i8<-127:127>:f32, 0.021174000576138496>>, %arg14: tensor<640x!quant.uniform<i32:f32, 1.601389680352559E-4>>, %arg15: tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, %arg16: tensor<2048x!quant.uniform<i16:f32, 1.1000000085914508E-4>>, %arg17: tensor<2048x!quant.uniform<i16:f32, 1.6799999866634607E-4>>, %arg18: tensor<2048x!quant.uniform<i16:f32, 1.55999994603917E-4>>, %arg19: tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>, %arg20: tensor<1x2048x!quant.uniform<i16:f32, 4.8799999058246613E-4>>) -> tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>> {
    %cst = "tfl.no_value"() {value = unit} : () -> none
    %0 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %cst, %cst, %cst, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg19, %arg20, %arg15, %arg16, %arg17, %arg18) ({}) {cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", input_to_input_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0049890000373125076>>, input_to_forget_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0078849997371435165>>, input_to_cell_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0087630003690719604>>, input_to_output_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0057529998011887074>>, effective_hidden_scale_intermediate = tensor<0x!quant.uniform<i8<-127:127>:f32, 0.0075630000792443752:2>>, kernel_type = #tfl<"lstm_kernel_type_attr FULL">, proj_clip = 0.01 : f32} : (tensor<1x528x!quant.uniform<i8:f32, 0.037248000502586365:-19>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.031925998628139496>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.056272000074386597>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.063763998448848724>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.013358999975025654>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.022830000147223473>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.032276000827550888>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.035427000373601913>>, none, none, none, tensor<2048x!quant.uniform<i32:f32, 4.2675782196965883E-7>>, tensor<2048x!quant.uniform<i32:f32, 1.0742187583900886E-7>>, tensor<2048x!quant.uniform<i32:f32, 1.6406249869760359E-7>>, tensor<2048x!quant.uniform<i32:f32, 1.523437447303877E-7>>, tensor<640x2048x!quant.uniform<i8<-127:127>:f32, 0.021174000576138496>>, tensor<640x!quant.uniform<i32:f32, 1.601389680352559E-4>>, tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>, tensor<1x2048x!quant.uniform<i16:f32, 4.8799999058246613E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, tensor<2048x!quant.uniform<i16:f32, 1.1000000085914508E-4>>, tensor<2048x!quant.uniform<i16:f32, 1.6799999866634607E-4>>, tensor<2048x!quant.uniform<i16:f32, 1.55999994603917E-4>>) -> tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>
  func.return %0 : tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>
// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:   operator_codes: [ {
// CHECK-NEXT:     deprecated_builtin_code: 16,
// CHECK-NEXT:     version: 1,
// CHECK-NEXT:     builtin_code: LSTM
// CHECK-NEXT:   } ],
// CHECK-NEXT:   subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 1, 528 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "arg0",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.037248 ],
// CHECK-NEXT:         zero_point: [ -19 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048, 528 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "arg1",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.059802 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048, 528 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "arg2",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.031926 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048, 528 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 4,
// CHECK-NEXT:       name: "arg3",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.056272 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048, 528 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 5,
// CHECK-NEXT:       name: "arg4",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.063764 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048, 640 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 6,
// CHECK-NEXT:       name: "arg5",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.013359 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048, 640 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 7,
// CHECK-NEXT:       name: "arg6",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.02283 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048, 640 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 8,
// CHECK-NEXT:       name: "arg7",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.032276 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048, 640 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 9,
// CHECK-NEXT:       name: "arg8",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.035427 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 10,
// CHECK-NEXT:       name: "arg9",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.0 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 11,
// CHECK-NEXT:       name: "arg10",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.0 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 12,
// CHECK-NEXT:       name: "arg11",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.0 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 13,
// CHECK-NEXT:       name: "arg12",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.0 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 640, 2048 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 14,
// CHECK-NEXT:       name: "arg13",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.021174 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 640 ],
// CHECK-NEXT:       type: INT32,
// CHECK-NEXT:       buffer: 15,
// CHECK-NEXT:       name: "arg14",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.00016 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048 ],
// CHECK-NEXT:       type: INT16,
// CHECK-NEXT:       buffer: 16,
// CHECK-NEXT:       name: "arg15",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.000437 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048 ],
// CHECK-NEXT:       type: INT16,
// CHECK-NEXT:       buffer: 17,
// CHECK-NEXT:       name: "arg16",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.00011 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048 ],
// CHECK-NEXT:       type: INT16,
// CHECK-NEXT:       buffer: 18,
// CHECK-NEXT:       name: "arg17",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.000168 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 2048 ],
// CHECK-NEXT:       type: INT16,
// CHECK-NEXT:       buffer: 19,
// CHECK-NEXT:       name: "arg18",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.000156 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 1, 640 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       name: "arg19",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.096711 ],
// CHECK-NEXT:         zero_point: [ 10 ]
// CHECK-NEXT:       },
// CHECK-NEXT:       is_variable: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 1, 2048 ],
// CHECK-NEXT:       type: INT16,
// CHECK-NEXT:       name: "arg20",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.000488 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       },
// CHECK-NEXT:       is_variable: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       type: INT16,
// CHECK-NEXT:       name: "input_to_input_intermediate",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.004989 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       type: INT16,
// CHECK-NEXT:       name: "input_to_forget_intermediate",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.007885 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       type: INT16,
// CHECK-NEXT:       name: "input_to_cell_intermediate",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.008763 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       type: INT16,
// CHECK-NEXT:       name: "input_to_output_intermediate",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.005753 ],
// CHECK-NEXT:         zero_point: [ 0 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 0 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       name: "effective_hidden_scale_intermediate",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.007563 ],
// CHECK-NEXT:         zero_point: [ 2 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 1, 640 ],
// CHECK-NEXT:       type: INT8,
// CHECK-NEXT:       buffer: 22,
// CHECK-NEXT:       name: "tfl.lstm",
// CHECK-NEXT:       quantization: {
// CHECK-NEXT:         scale: [ 0.096711 ],
// CHECK-NEXT:         zero_point: [ 10 ]
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ],
// CHECK-NEXT:     outputs: [ 26 ],
// CHECK-NEXT:     operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, 9, 10, 11, 12, 13, 14, 19, 20, 15, 16, 17, 18 ],
// CHECK-NEXT:       outputs: [ 26 ],
// CHECK-NEXT:       builtin_options_type: LSTMOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-NEXT:         fused_activation_function: TANH,
// CHECK-NEXT:         cell_clip: 10.0,
// CHECK-NEXT:         proj_clip: 0.01
// CHECK-NEXT:       },
// CHECK-NEXT:       intermediates: [ 21, 22, 23, 24, 25 ]
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
// CHECK-NEXT:     data: [ 49, 46, 55, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   } ],
// CHECK-NEXT:   metadata: [ {
// CHECK-NEXT:     name: "min_runtime_version",
// CHECK-NEXT:     buffer: 23
// CHECK-NEXT:   } ]
// CHECK-NEXT:   signature_defs: [ ]
// CHECK-NEXT: }
}
