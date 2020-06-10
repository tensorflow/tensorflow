// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s

// This test is to test for unranked function output from input, the output type should be compatible with input type.

// CHECK: func @main(%arg0: tensor<1xf32>) -> tensor<*xf32>
// CHECK: %0 = "tf.While"(%arg0) {body = @body, cond = @cond, is_stateless = false} : (tensor<1xf32>) -> tensor<*xf32>
// CHECK: return %0 : tensor<*xf32>
// CHECK: func @cond(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK: func @body(%arg0: tensor<*xf32>) -> tensor<*xf32>

func @main(%arg0: tensor<1xf32>) -> tensor<*xf32> {
  %0 = "tf.While"(%arg0) {cond = @cond, body = @body, is_stateless = false} : (tensor<1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @cond(%arg1: tensor<*xf32>) -> tensor<*xf32> {
  return %arg1: tensor<*xf32>
}

func @body(%arg1: tensor<*xf32>) -> tensor<*xf32> {
  return %arg1: tensor<*xf32>
}
