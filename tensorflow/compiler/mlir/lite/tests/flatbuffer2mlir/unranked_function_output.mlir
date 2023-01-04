// RUN: tf-opt --tfl-legalize-tf-while %s -o - | flatbuffer_translate -mlir-to-tflite-flatbuffer - -o -  | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// This test is to test for unranked function output from input, the output type should be compatible with input type.

// CHECK: func @main(%arg0: tensor<1xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.while"}} {
// CHECK:   %0 = "tfl.while"(%arg0) ({
// CHECK:   ^bb0(%arg1: tensor<*xf32>):
// CHECK:     %[[RES0:.*]] = func.call @cond(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:     "tfl.yield"(%[[RES0]]) : (tensor<*xf32>) -> ()
// CHECK:   },  {
// CHECK:   ^bb0(%arg1: tensor<*xf32>):
// CHECK:     %[[RES1:.*]] = func.call @body(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:     "tfl.yield"(%[[RES1]]) : (tensor<*xf32>) -> ()
// CHECK:   }) : (tensor<1xf32>) -> tensor<*xf32>
// CHECK:   return %0 : tensor<*xf32>
// CHECK: }
func.func @main(%arg0: tensor<1xf32>) -> tensor<*xf32> {
  %0 = "tf.While"(%arg0) {cond = @cond, body = @body, is_stateless = false} : (tensor<1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @cond(%arg1: tensor<*xf32>) -> tensor<*xf32> {
  func.return %arg1: tensor<*xf32>
}

func.func @body(%arg1: tensor<*xf32>) -> tensor<*xf32> {
  func.return %arg1: tensor<*xf32>
}
