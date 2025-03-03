// RUN: xla-translate -mlir-hlo-to-hlo-text %s | FileCheck %s
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-use-tuple-args -emit-return-tuple %s | FileCheck %s --check-prefix=TUPLE

// Test to verify that multiple result function with always emit return tuple
// does not result in nested tuples.

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: s32[4]) -> (s32[4], s32[1,2,3,4])
// TUPLE-LABEL: ENTRY %main.{{.*}} (arg_tuple.1: (s32[4])) -> (s32[4], s32[1,2,3,4])
func.func @main(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<1x2x3x4xi32>) {
  // CHECK-NEXT: %Arg_0.1 = s32[4] parameter(0)
  // CHECK-NEXT: %broadcast.2 = s32[1,2,3,4] broadcast(%Arg_0.1), dimensions={3}
  %0 = "mhlo.broadcast"(%arg0) <{broadcast_sizes = dense<[1,2,3]> : tensor<3xi64>}> : (tensor<4xi32>) -> tensor<1x2x3x4xi32>
  func.return %arg0, %0 : tensor<4xi32>, tensor<1x2x3x4xi32>
}
