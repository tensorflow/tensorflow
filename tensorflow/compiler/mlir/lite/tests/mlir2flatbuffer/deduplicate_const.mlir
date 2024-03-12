// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

module {
func.func @add(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {tf.entry_function = {inputs = "serving_default_x", outputs = "outputs"}} {
  %0 = "tfl.pseudo_const" () {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  %1 = "tfl.add" (%0, %arg0) {fused_activation_function = "NONE"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  func.return %1 : tensor<3x2xf32>
}

func.func @sub(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {tf.entry_function = {inputs = "serving_default_x", outputs = "outputs"}} {
  %0 = "tfl.pseudo_const" () {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  %1 = "tfl.sub" (%0, %arg0) {fused_activation_function = "NONE"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  func.return %1 : tensor<3x2xf32>
}
}

// CHECK:      {
// CHECK:        subgraphs: [ {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 1,
// CHECK-NEXT:       name: "serving_default_x",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "tfl.pseudo_const",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 3,
// CHECK-NEXT:       name: "outputs",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     } ],
// CHECK:          name: "add"
// CHECK-NEXT:   }, {
// CHECK-NEXT:     tensors: [ {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 4,
// CHECK-NEXT:       name: "serving_default_x",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 2,
// CHECK-NEXT:       name: "tfl.pseudo_const1",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     }, {
// CHECK-NEXT:       shape: [ 3, 2 ],
// CHECK-NEXT:       buffer: 6,
// CHECK-NEXT:       name: "outputs",
// CHECK-NEXT:       quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:       },
// CHECK-NEXT:       has_rank: true
// CHECK-NEXT:     } ],
// CHECK-NEXT:     inputs: [ 0 ],
// CHECK-NEXT:     outputs: [ 2 ],
// CHECK:          name: "sub"
// CHECK-NEXT:   } ],
// CHECK-NEXT:   description: "MLIR Converted.",
// CHECK-NEXT:   buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64 ]
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 49, 46, 54, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:   } ],
// CHECK:      }