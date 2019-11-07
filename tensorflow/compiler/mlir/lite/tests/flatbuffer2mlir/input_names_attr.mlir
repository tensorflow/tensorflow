// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s

// Tests input names from FlatBuffer are added to `tf.entry_function` attribute.

// CHECK-LABEL: @main
func @main(%arg0: tensor<4xi8>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi8>)
// CHECK: attributes {tf.entry_function = {inputs = "input0,input1"}}
attributes {tf.entry_function = {inputs = "input0,input1", outputs = "output0,output1"}} {
  return %arg1, %arg0 : tensor<4xi32>, tensor<4xi8>
}
