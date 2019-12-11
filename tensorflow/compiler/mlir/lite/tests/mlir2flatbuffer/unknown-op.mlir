// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - 2>&1 | FileCheck %s; test ${PIPESTATUS[0]} -ne 0

func @main(tensor<3x2xi32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<3x2xi32>):
  // CHECK: error: 'unknown_op' op dialect is not registered
  %0 = "unknown_op"(%arg0) : (tensor<3x2xi32>) -> tensor<3x2xi32>
  return %0 : tensor<3x2xi32>
}
