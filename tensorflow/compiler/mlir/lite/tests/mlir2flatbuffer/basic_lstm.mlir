// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<1x384xf32>, tensor<1x96xf32>, tensor<384x480xf32>, tensor<384xf32>, tensor<1x96xf32>) -> tensor<1x96xf32> {
// CHECK: {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    deprecated_builtin_code: 16,
// CHECK-NEXT:    version: 2
// CHECK-NEXT:    builtin_code: LSTM
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 1, 384 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 96 ],
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "arg1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 384, 480 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "arg2",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 384 ],
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "arg3",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 96 ],
// CHECK-NEXT:      buffer: 5,
// CHECK-NEXT:      name: "arg4",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 96 ],
// CHECK-NEXT:      buffer: 6,
// CHECK-NEXT:      name: "tfl.basic_lstm",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 96 ],
// CHECK-NEXT:      buffer: 7,
// CHECK-NEXT:      name: "tfl.basic_lstm:1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 480 ],
// CHECK-NEXT:      buffer: 8,
// CHECK-NEXT:      name: "tfl.basic_lstm:2",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 384 ],
// CHECK-NEXT:      buffer: 9,
// CHECK-NEXT:      name: "tfl.basic_lstm:3",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1, 2, 3, 4 ],
// CHECK-NEXT:    outputs: [ 5 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 1, 2, 3, 4 ],
// CHECK-NEXT:      outputs: [ 5, 6, 7, 8 ],
// CHECK-NEXT:      builtin_options_type: LSTMOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-NEXT:        fused_activation_function: RELU,
// CHECK-NEXT:        cell_clip: 1.0,
// CHECK-NEXT:        proj_clip: 2.0,
// CHECK-NEXT:        kernel_type: BASIC
// CHECK-NEXT:      },
// CHECK-NEXT:      intermediates: [ ]
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:  data: [ 49, 46, 49, 48, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:  name: "min_runtime_version",
// CHECK-NEXT:  buffer: 10
// CHECK-NEXT:  } ]
// CHECK-NEXT:  signature_defs: [ ]
// CHECK-NEXT:}

^bb0(%arg0: tensor<1x384xf32>, %arg1: tensor<1x96xf32>, %arg2: tensor<384x480xf32>, %arg3: tensor<384xf32>, %arg4: tensor<1x96xf32>):
  %0:4 = "tfl.basic_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4) {fused_activation_function = "RELU", cell_clip = 1.0 : f32, proj_clip = 2.0 : f32} : (tensor<1x384xf32>, tensor<1x96xf32>, tensor<384x480xf32>, tensor<384xf32>, tensor<1x96xf32>) -> (tensor<1x96xf32>, tensor<1x96xf32>, tensor<1x480xf32>, tensor<1x384xf32>)
  return %0#0 : tensor<1x96xf32>
}
