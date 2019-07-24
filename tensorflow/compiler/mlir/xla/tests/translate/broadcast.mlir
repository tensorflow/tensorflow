// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main.3 (Arg_0.1: s32[4]) -> s32[1,2,3,4] {
func @main(%arg0: tensor<4xi32>) -> tensor<1x2x3x4xi32> {
  // CHECK-NEXT: %Arg_0.1 = s32[4] parameter(0)
  // CHECK-NEXT: ROOT %broadcast.2 = s32[1,2,3,4] broadcast(s32[4] %Arg_0.1), dimensions={3}
  %0 = "xla.broadcast"(%arg0) {broadcast_sizes = dense<[1,2,3]> : tensor<3xi64>} : (tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %0 : tensor<1x2x3x4xi32>
}
