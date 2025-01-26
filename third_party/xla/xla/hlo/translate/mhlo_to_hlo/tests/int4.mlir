// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// Test int4 constants and conversions.

// CHECK-LABEL: ENTRY %main.{{.*}} () -> s4[6]
func.func @main() -> tensor<6xi4> {
  // CHECK-NEXT: %[[CONSTANT:.*]] = s4[6] constant({1, -2, -3, 4, -8, 7})
  %0 = mhlo.constant dense<[1, -2, -3, 4, -8, 7]> : tensor<6xi4>
  // CHECK-NEXT: %[[CONVERT1:.*]] = s8[6] convert(s4[6] %[[CONSTANT]])
  %1 = "mhlo.convert"(%0) : (tensor<6xi4>) -> tensor<6xi8>
  // CHECK-NEXT: ROOT %[[CONVERT2:.*]] = s4[6] convert(s8[6] %[[CONVERT1]])
  %2 = "mhlo.convert"(%1) : (tensor<6xi8>) -> tensor<6xi4>
  func.return %2 : tensor<6xi4>
}

// -----

// CHECK-LABEL: ENTRY %main.{{.*}} () -> u4[4]
func.func @main() -> tensor<4xui4> {
  // CHECK-NEXT: %[[CONSTANT:.*]] = u4[4] constant({1, 2, 3, 15})
  %0 = mhlo.constant dense<[1, 2, 3, 15]> : tensor<4xui4>
  // CHECK-NEXT: %[[CONVERT1:.*]] = u8[4] convert(u4[4] %[[CONSTANT]])
  %1 = "mhlo.convert"(%0) : (tensor<4xui4>) -> tensor<4xui8>
  // CHECK-NEXT: ROOT %[[CONVERT2:.*]] = u4[4] convert(u8[4] %[[CONVERT1]])
  %2 = "mhlo.convert"(%1) : (tensor<4xui8>) -> tensor<4xui4>
  func.return %2 : tensor<4xui4>
}
