// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// This test is to test for unranked function output from input, the output type should be compatible with input type.

// CHECK: func @main(%arg0: tensor<1xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tf.While"}} {
// CHECK:   %0 = "tfl.while"(%arg0) ( {
// CHECK:   ^bb0(%arg1: tensor<*xf32>):
// CHECK:     %[[RES0:.*]] = call @cond(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:     "tfl.yield"(%[[RES0]]) : (tensor<*xf32>) -> ()
// CHECK:   },  {
// CHECK:   ^bb0(%arg1: tensor<*xf32>):
// CHECK:     %[[RES1:.*]] = call @body(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:     "tfl.yield"(%[[RES1]]) : (tensor<*xf32>) -> ()
// CHECK:   }) : (tensor<1xf32>) -> tensor<*xf32>
// CHECK:   return %0 : tensor<*xf32>
// CHECK: }
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
