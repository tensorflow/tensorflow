// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main
func @main(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  // CHECK-NEXT: %Arg_0.1 = s32[1,2,3,4] parameter(0)

  // CHECK-NEXT: ROOT %transpose.2 = s32[2,1,4,3] transpose(s32[1,2,3,4] %Arg_0.1), dimensions={1,0,3,2}
  %0 = "xla.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0 : tensor<2x1x4x3xi32>
}

