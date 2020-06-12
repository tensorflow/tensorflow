// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1001xf32> {
// CHECK: {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    builtin_code: QUANTIZE,
// CHECK-NEXT:    version: 1
// CHECK-NEXT:  }, {
// CHECK-NEXT:    builtin_code: CONV_2D,
// CHECK-NEXT:    version: 1
// CHECK-NEXT:  }, {
// CHECK-NEXT:    builtin_code: RESHAPE,
// CHECK-NEXT:    version: 1
// CHECK-NEXT:  }, {
// CHECK-NEXT:    builtin_code: SOFTMAX,
// CHECK-NEXT:    version: 1
// CHECK-NEXT:  }, {
// CHECK-NEXT:    builtin_code: DEQUANTIZE,
// CHECK-NEXT:    version: 1
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 1, 224, 224, 3 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 2 ],
// CHECK-NEXT:      type: INT32,
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "Const",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 224, 224, 3 ],
// CHECK-NEXT:      type: UINT8,
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "tfl.quantize",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 0.007812 ],
// CHECK-NEXT:        zero_point: [ 128 ]
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 32, 3, 3, 3 ],
// CHECK-NEXT:      type: UINT8,
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "tfl.pseudo_qconst",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 0.021827 ],
// CHECK-NEXT:        zero_point: [ 151 ]
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 32 ],
// CHECK-NEXT:      type: INT32,
// CHECK-NEXT:      buffer: 5,
// CHECK-NEXT:      name: "tfl.pseudo_qconst1",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 0.000171 ],
// CHECK-NEXT:        zero_point: [ 0 ]
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 112, 112, 32 ],
// CHECK-NEXT:      type: UINT8,
// CHECK-NEXT:      buffer: 6,
// CHECK-NEXT:      name: "tfl.conv_2d",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 0.023528 ],
// CHECK-NEXT:        zero_point: [ 0 ]
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 1001 ],
// CHECK-NEXT:      type: UINT8,
// CHECK-NEXT:      buffer: 7,
// CHECK-NEXT:      name: "tfl.reshape",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 0.023528 ],
// CHECK-NEXT:        zero_point: [ 0 ]
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 1001 ],
// CHECK-NEXT:      type: UINT8,
// CHECK-NEXT:      buffer: 8,
// CHECK-NEXT:      name: "tfl.softmax",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 0.003906 ],
// CHECK-NEXT:        zero_point: [ 0 ]
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 1001 ],
// CHECK-NEXT:      buffer: 9,
// CHECK-NEXT:      name: "tfl.dequantize",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0 ],
// CHECK-NEXT:    outputs: [ 8 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0 ],
// CHECK-NEXT:      outputs: [ 2 ]
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 1,
// CHECK-NEXT:      inputs: [ 2, 3, 4 ],
// CHECK-NEXT:      outputs: [ 5 ],
// CHECK-NEXT:      builtin_options_type: Conv2DOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-NEXT:        stride_w: 5,
// CHECK-NEXT:        stride_h: 4,
// CHECK-NEXT:        dilation_w_factor: 3,
// CHECK-NEXT:        dilation_h_factor: 2
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 2,
// CHECK-NEXT:      inputs: [ 5, 1 ],
// CHECK-NEXT:      outputs: [ 6 ]
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 3,
// CHECK-NEXT:      inputs: [ 6 ],
// CHECK-NEXT:      outputs: [ 7 ],
// CHECK-NEXT:      builtin_options_type: SoftmaxOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-NEXT:        beta: 1.0
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 4,
// CHECK-NEXT:      inputs: [ 7 ],
// CHECK-NEXT:      outputs: [ 8 ]
// CHECK-NEXT:    } ]
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:      data: [ 1, 0, 0, 0, 233, 3, 0, 0 ]
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180 ]
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 49, 46, 49, 51, 46, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:  name: "min_runtime_version",
// CHECK-NEXT:  buffer: 10
// CHECK-NEXT:  } ]
// CHECK-NEXT:}

  %0 = "tfl.pseudo_const" () {value = dense<[1, 1001]> : tensor<2xi32>} : () -> tensor<2xi32> loc("Const")
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %4 = "tfl.conv_2d"(%1, %2, %3) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %5 = "tfl.reshape"(%4, %0) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x1001x!quant.uniform<u8:f32, 0.023528476789885875>>
  %6 = "tfl.softmax"(%5) {beta = 1.000000e+00 : f32} : (tensor<1x1001x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x1001x!quant.uniform<u8:f32, 3.906250e-03>>
  %7 = "tfl.dequantize"(%6) : (tensor<1x1001x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<1x1001xf32>
  return %7 : tensor<1x1001xf32>
}
