// RUN: sdy_opt %s -xla-sdy-round-trip-testing-pipeline -split-input-file 2>&1 | FileCheck %s

// Basic test with non flat graphs. This test takes an SDY module, exports it to
// StableHLO while saving the SDY attrs and meshes, goes to HLO, back to
// StableHLO, and then back to SDY.

// CHECK: sdy.mesh @mesh = <["a"=2]>
sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func @main(
// CHECK-SAME: %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xf32> {
  // CHECK: sdy.named_computation<"foo.8">
  %0 = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
    // CHECK: sdy.named_computation<"bar.6">
    %2 = sdy.named_computation<"bar">(%1) (%arg2: tensor<8xf32>) {
      %3 = stablehlo.abs %arg2 : tensor<8xf32>
      sdy.return %3 : tensor<8xf32>
    } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
  %4 = stablehlo.negate %0 : tensor<8xf32>
  // CHECK: sdy.named_computation<"baz.13">
  %5 = sdy.named_computation<"baz">(%4) (%arg1: tensor<8xf32>) {
    %6 = stablehlo.abs %arg1 : tensor<8xf32>
    sdy.return %6 : tensor<8xf32>
  } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK: sdy.named_computation<"bar_0.17">
  %7 = sdy.named_computation<"bar">(%5) (%arg1: tensor<8xf32>) {
    %8 = stablehlo.abs %arg1 : tensor<8xf32>
    sdy.return %8 : tensor<8xf32>
  } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %7 : tensor<8xf32>
}
