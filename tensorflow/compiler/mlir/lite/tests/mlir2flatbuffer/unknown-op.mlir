// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - 2>&1 | FileCheck %s; if [[ ${PIPESTATUS[0]} != 0  &&  ${PIPESTATUS[1]} == 0 ]]; then exit 0; else exit 1; fi

func @main(tensor<3x2xi32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<3x2xi32>):
  // CHECK: error: 'unknown_op' op dialect is not registered
  %0 = "unknown_op"(%arg0) : (tensor<3x2xi32>) -> tensor<3x2xi32>
  return %0 : tensor<3x2xi32>
}
