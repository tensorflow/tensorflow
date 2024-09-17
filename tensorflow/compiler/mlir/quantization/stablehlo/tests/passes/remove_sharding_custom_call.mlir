// RUN: stablehlo-quant-opt %s -stablehlo-remove-sharding-custom-call \
// RUN:   -split-input-file | FileCheck %s

// CHECK-LABEL: sharding_custom_call_removed
func.func @sharding_custom_call_removed(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %1 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = ""} : (tensor<3xf32>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
}
// CHECK-NOT: custom_call

// -----

// Tests that a custom_call that is not @Sharding is not removed.

// CHECK-LABEL: custom_call_not_removed
func.func @custom_call_not_removed(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %1 = stablehlo.custom_call @NotSharding(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
}
// CHECK: custom_call @NotSharding
