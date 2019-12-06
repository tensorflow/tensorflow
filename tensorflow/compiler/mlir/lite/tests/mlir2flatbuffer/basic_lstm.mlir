// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck --dump-input-on-failure %s

func @main(tensor<1x384xf32>, tensor<1x96xf32>, tensor<384x480xf32>, tensor<384xf32>, tensor<1x96xf32>) -> tensor<1x96xf32> {
// CHECK: {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    builtin_code: LSTM,
// CHECK-NEXT:    version: 2
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 1, 384 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "tfl.pseudo_input",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 96 ],
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "tfl.pseudo_input1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 384, 480 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "tfl.pseudo_input2",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 384 ],
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "tfl.pseudo_input3",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 96 ],
// CHECK-NEXT:      buffer: 5,
// CHECK-NEXT:      name: "tfl.pseudo_input4",
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
// CHECK-NEXT:      }
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
// CHECK-NEXT:  } ]
// CHECK-NEXT:}

^bb0(%arg0: tensor<1x384xf32>, %arg1: tensor<1x96xf32>, %arg2: tensor<384x480xf32>, %arg3: tensor<384xf32>, %arg4: tensor<1x96xf32>):
  %0 = "tfl.pseudo_input" (%arg0) : (tensor<1x384xf32>) -> tensor<1x384xf32>
  %1 = "tfl.pseudo_input" (%arg1) : (tensor<1x96xf32>) -> tensor<1x96xf32>
  %2 = "tfl.pseudo_input" (%arg2) : (tensor<384x480xf32>) -> tensor<384x480xf32>
  %3 = "tfl.pseudo_input" (%arg3) : (tensor<384xf32>) -> tensor<384xf32>
  %4 = "tfl.pseudo_input" (%arg4) : (tensor<1x96xf32>) -> tensor<1x96xf32>
  %5:4 = "tfl.basic_lstm"(%0, %1, %2, %3, %4) {fused_activation_function = "RELU", cell_clip = 1.0 : f32, proj_clip = 2.0 : f32} : (tensor<1x384xf32>, tensor<1x96xf32>, tensor<384x480xf32>, tensor<384xf32>, tensor<1x96xf32>) -> (tensor<1x96xf32>, tensor<1x96xf32>, tensor<1x480xf32>, tensor<1x384xf32>)
  return %5#0 : tensor<1x96xf32>
}
