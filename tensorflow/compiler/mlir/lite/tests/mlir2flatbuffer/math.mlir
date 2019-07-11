// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck --dump-input-on-failure %s

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
  // CHECK:      {
  // CHECK-NEXT:   version: 3,
  // CHECK-NEXT:   operator_codes: [ {
  // CHECK-NEXT:     builtin_code: SQUARED_DIFFERENCE
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     builtin_code: MUL
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     builtin_code: DIV
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     builtin_code: EXP
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     builtin_code: NEG
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   subgraphs: [ {
  // CHECK-NEXT:     tensors: [ {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       buffer: 1,
  // CHECK-NEXT:       name: "Input",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ 4 ],
  // CHECK-NEXT:       buffer: 2,
  // CHECK-NEXT:       name: "Const",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ ],
  // CHECK-NEXT:       buffer: 3,
  // CHECK-NEXT:       name: "squared_difference",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ ],
  // CHECK-NEXT:       buffer: 4,
  // CHECK-NEXT:       name: "mul",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ ],
  // CHECK-NEXT:       buffer: 5,
  // CHECK-NEXT:       name: "div",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ ],
  // CHECK-NEXT:       buffer: 6,
  // CHECK-NEXT:       name: "exp",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       shape: [ ],
  // CHECK-NEXT:       buffer: 7,
  // CHECK-NEXT:       name: "neg",
  // CHECK-NEXT:       quantization: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     } ],
  // CHECK-NEXT:     inputs: [ 0 ],
  // CHECK-NEXT:     outputs: [ 6 ],
  // CHECK-NEXT:     operators: [ {
  // CHECK-NEXT:       inputs: [ 0, 1 ],
  // CHECK-NEXT:       outputs: [ 2 ]
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       opcode_index: 1,
  // CHECK-NEXT:       inputs: [ 0, 2 ],
  // CHECK-NEXT:       outputs: [ 3 ],
  // CHECK-NEXT:       builtin_options_type: MulOptions,
  // CHECK-NEXT:       builtin_options: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       opcode_index: 2,
  // CHECK-NEXT:       inputs: [ 3, 2 ],
  // CHECK-NEXT:       outputs: [ 4 ],
  // CHECK-NEXT:       builtin_options_type: DivOptions,
  // CHECK-NEXT:       builtin_options: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       opcode_index: 3,
  // CHECK-NEXT:       inputs: [ 4 ],
  // CHECK-NEXT:       outputs: [ 5 ],
  // CHECK-NEXT:       builtin_options_type: ExpOptions,
  // CHECK-NEXT:       builtin_options: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }, {
  // CHECK-NEXT:       opcode_index: 4,
  // CHECK-NEXT:       inputs: [ 5 ],
  // CHECK-NEXT:       outputs: [ 6 ],
  // CHECK-NEXT:       builtin_options_type: NegOptions,
  // CHECK-NEXT:       builtin_options: {
  // CHECK-EMPTY:
  // CHECK-NEXT:       }
  // CHECK-NEXT:     } ]
  // CHECK-NEXT:    name: "main"
  // CHECK-NEXT:   } ],
  // CHECK-NEXT:   description: "MLIR Converted.",
  // CHECK-NEXT:   buffers: [ {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-EMPTY:
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:     data: [ 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63 ]
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

  %0 = "tfl.pseudo_input" (%arg0) : (tensor<4xf32>) -> tensor<4xf32> loc("Input")
  %1 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %2 = "tfl.squared_difference"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("squared_difference")
  %3 = "tfl.mul"(%0, %2) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("mul")
  %4 = "tfl.div"(%3, %2) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("div")
  %5 = "tfl.exp"(%4) : (tensor<4xf32>) -> tensor<4xf32> loc("exp")
  %6 = "tfl.neg"(%5) : (tensor<4xf32>) -> tensor<4xf32> loc("neg")
  return %6 : tensor<4xf32>
}
