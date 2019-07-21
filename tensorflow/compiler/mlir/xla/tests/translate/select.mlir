// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main
func @main(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-NEXT: %Arg_0.1 = pred[2,3] parameter(0)
  // CHECK-NEXT: %Arg_1.2 = s32[2,3] parameter(1)
  // CHECK-NEXT: %Arg_2.3 = s32[2,3] parameter(2)

  // CHECK-NEXT: ROOT %select.4 = s32[2,3] select(pred[2,3] %Arg_0.1, s32[2,3] %Arg_1.2, s32[2,3] %Arg_2.3)
  %0 = "xla.select"(%arg0, %arg1, %arg2) {name = "select.4"} : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

