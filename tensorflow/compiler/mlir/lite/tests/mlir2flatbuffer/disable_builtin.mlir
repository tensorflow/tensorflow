// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-builtin-tflite-ops=false -o - | flatbuffer_to_string - | FileCheck %s; test ${PIPESTATUS[1]} -eq 1
# CHECK: loc("disable_builtin.mlir":2:1): is a TFLite builtin op but builtin emission is not enabled
# CHECK-NEXT: Verification failed.

func @main(tensor<3x2xi32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<3x2xi32>):
  %0 = "std.constant" () {name = "Const2", value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add" (%0, %1) {name = "add"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32>
  return %1 : tensor<3x2xi32>
}
