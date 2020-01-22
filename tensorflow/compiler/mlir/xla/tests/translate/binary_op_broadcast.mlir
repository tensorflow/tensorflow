// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main.13 (Arg_0.1: s32[1,4], Arg_1.2: s32[2,4], Arg_2.3: s32[2,3,4]) -> s32[2,3,4] {
func @main(%arg0: tensor<1x4xi32>, %arg1: tensor<2x4xi32>, %arg2: tensor<2x3x4xi32>) -> tensor<2x3x4xi32> {
  // Same rank degenerate broadcast
  // CHECK-NEXT: %Arg_0.1 = s32[1,4] parameter(0)
  // CHECK-NEXT: %reshape.4 = s32[4] reshape(s32[1,4] %Arg_0.1)
  // CHECK-NEXT: %broadcast.5 = s32[2,4] broadcast(s32[4] %reshape.4)
  // CHECK-NEXT: %Arg_1.2 = s32[2,4] parameter(1)
  // CHECK-NEXT: %add.6 = s32[2,4] add(s32[2,4] %broadcast.5, s32[2,4] %Arg_1.2)
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<1x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>

  // Broadcast up rank
  // CHECK-NEXT: %broadcast.7 = s32[2,3,4] broadcast(s32[2,4] %Arg_1.2), dimensions={0,2}
  // CHECK-NEXT: %Arg_2.3 = s32[2,3,4] parameter(2)
  // CHECK-NEXT: %add.8 = s32[2,3,4] add(s32[2,3,4] %broadcast.7, s32[2,3,4] %Arg_2.3)
  %1 = "xla_hlo.add"(%arg1, %arg2) {broadcast_dimensions = dense<[0,2]> : tensor<2xi64>} : (tensor<2x4xi32>, tensor<2x3x4xi32>) -> tensor<2x3x4xi32>

  // Broadcast up rank + degenerate broadcast
  // CHECK-NEXT: %broadcast.9 = s32[2,1,4] broadcast(s32[1,4] %Arg_0.1), dimensions={1,2}
  // CHECK-NEXT: %reshape.10 = s32[2,4] reshape(s32[2,1,4] %broadcast.9)
  // CHECK-NEXT: %broadcast.11 = s32[2,3,4] broadcast(s32[2,4] %reshape.10), dimensions={0,2}
  // CHECK-NEXT: ROOT %add.12 = s32[2,3,4] add(s32[2,3,4] %broadcast.11, s32[2,3,4] %Arg_2.3)
  %2 = "xla_hlo.add"(%arg0, %arg2) {broadcast_dimensions = dense<[1,2]> : tensor<2xi64>} : (tensor<1x4xi32>, tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
  return %2 : tensor<2x3x4xi32>
}
