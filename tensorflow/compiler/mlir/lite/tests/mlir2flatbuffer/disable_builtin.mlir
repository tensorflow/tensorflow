// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer -emit-builtin-tflite-ops=false %s 2>&1 | FileCheck %s; if [[ ${PIPESTATUS[0]} != 0  &&  ${PIPESTATUS[1]} == 0 ]]; then exit 0; else exit 1; fi

// CHECK: 'tfl.add' op is a TFLite builtin op but builtin emission is not enabled

func @main(tensor<3x2xi32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<3x2xi32>):
  %0 = "std.constant"() {name = "Const2", value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.add"(%0, %arg0) {fused_activation_function = "NONE", name = "add"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32>
  return %1 : tensor<3x2xi32>
}
