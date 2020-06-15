// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

module {
func @serving_default(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> attributes {tf.entry_function = {inputs = "serving_default_x", outputs = "outputs"}} {
// CHECK:       {

// CHECK-LABEL:   version: 3,

// CHECK-LABEL:   operator_codes: [ {
// CHECK:           version: 1
// CHECK:         } ],

// CHECK-LABEL:   subgraphs: [ {
// CHECK:           tensors: [ {
// CHECK:             shape: [ 3, 2 ],
// CHECK:             buffer: 1,
// CHECK:             name: "serving_default_x",
// CHECK:             quantization: {
// CHECK:             }
// CHECK:           }, {
// CHECK:             shape: [ 3, 2 ],
// CHECK:             buffer: 2,
// CHECK:             name: "tfl.pseudo_const",
// CHECK:             quantization: {
// CHECK:             }
// CHECK:           }, {
// CHECK:             shape: [ 3, 2 ],
// CHECK:             buffer: 3,
// CHECK:             name: "outputs",
// CHECK:             quantization: {
// CHECK:             }
// CHECK:           } ],
// CHECK:           inputs: [ 0 ],
// CHECK:           outputs: [ 2 ],
// CHECK:           operators: [ {
// CHECK:             inputs: [ 1, 0 ],
// CHECK:             outputs: [ 2 ],
// CHECK:             builtin_options_type: AddOptions,
// CHECK:             builtin_options: {
// CHECK:             }
// CHECK:           } ],
// CHECK:           name: "main"
// CHECK:         } ],
// CHECK-LABEL:   description: "MLIR Converted.",
// CHECK-LABEL:   buffers: [ {
// CHECK:         }, {
// CHECK:         }, {
// CHECK:           data: [ 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64 ]
// CHECK:         }, {
// CHECK:         } ]
// CHECK:       }
  %0 = "tfl.pseudo_const" () {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  %1 = "tfl.add" (%0, %arg0) {fused_activation_function = "NONE"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}
}
