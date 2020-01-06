// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-custom-ops -o - | flatbuffer_to_string - | FileCheck %s

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):

// CHECK:  {
// CHECK-NEXT:    version: 3,
// CHECK-NEXT:    operator_codes: [ {
// CHECK-NEXT:      builtin_code: MUL,
// CHECK-NEXT:      version: 1
// CHECK-NEXT:    }, {
// CHECK-NEXT:      builtin_code: CUSTOM,
// CHECK-NEXT:      custom_code: "MyCustomOp"
// CHECK-NEXT:    }, {
// CHECK-NEXT:      builtin_code: EXP,
// CHECK-NEXT:      version: 1
// CHECK-NEXT:    } ],
// CHECK-NEXT:    subgraphs: [ {
// CHECK-NEXT:      tensors: [ {
// CHECK-NEXT:        shape: [ 4 ],
// CHECK-NEXT:        buffer: 1,
// CHECK-NEXT:        name: "arg0",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:      }, {
// CHECK-NEXT:        shape: [ 4 ],
// CHECK-NEXT:        buffer: 2,
// CHECK-NEXT:        name: "Const",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "mul",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "MyCustomOp",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 5,
// CHECK-NEXT:      name: "exp",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0 ],
// CHECK-NEXT:    outputs: [ 4 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 1 ],
// CHECK-NEXT:      outputs: [ 2 ],
// CHECK-NEXT:      builtin_options_type: MulOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 1,
// CHECK-NEXT:      inputs: [ 2, 1 ],
// CHECK-NEXT:      outputs: [ 3 ],
// CHECK-NEXT:      custom_options: [ 105, 110, 116, 95, 97, 116, 116, 114, 0, 102, 117, 115, 101, 100, 95, 97, 99, 116, 105, 118, 97, 116, 105, 111, 110, 95, 102, 117, 110, 99, 116, 105, 111, 110, 0, 4, 82, 69, 76, 85, 0, 2, 33, 43, 2, 1, 2, 11, 2, 20, 4, 4, 36, 1 ]
// CHECK-NEXT:    }, {
// CHECK-NEXT:      opcode_index: 2,
// CHECK-NEXT:      inputs: [ 3 ],
// CHECK-NEXT:      outputs: [ 4 ],
// CHECK-NEXT:      builtin_options_type: ExpOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-EMPTY:
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
// CHECK-NEXT:    data: [ 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63 ]
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  } ]
// CHECK-NEXT:}

  %0 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("mul")
  // tf.MyCustomOp is the result of conversion to a Custom op
  %2 = "tf.MyCustomOp"(%1, %0) {fused_activation_function = "RELU", int_attr = 2 : i32}  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("MyCustomOp")
  %3 = "tfl.exp"(%2)  : (tensor<4xf32>) -> tensor<4xf32> loc("exp")
  return %3 : tensor<4xf32>
}
