// RUN: sdy_opt %s -split-input-file -xla-sdy-stablehlo-round-trip-shard-map-export="erase-manual-computations=true" 2>&1 | FileCheck %s

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @manual_comp_erase
// CHECK-SAME: %[[ARG0:.*]]: tensor<8xf32>
func.func @manual_comp_erase(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[COPY_IN:.*]] = mhlo.copy %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @xla.sdy.inlineable_callee(%[[COPY_IN]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[COPY_OUT:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<8xf32>
  // CHECK-NEXT: return %[[COPY_OUT]] : tensor<8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @xla.sdy.inlineable_callee(
// CHECK-SAME: %[[ARG:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ARG]], %[[ARG]] : tensor<8xf32>
// CHECK-NEXT: return %[[ADD]] : tensor<8xf32>

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @manual_comp_erase_with_users_and_uses
// CHECK-SAME: %[[ARG0:.*]]: tensor<8xf32>
func.func @manual_comp_erase_with_users_and_uses(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE_IN:.*]] = stablehlo.negate %[[ARG0]] : tensor<8xf32>
  // CHECK-NEXT: %[[COPY_IN:.*]] = mhlo.copy %[[NEGATE_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @xla.sdy.inlineable_callee(%[[COPY_IN]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[COPY_OUT:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE_OUT:.*]] = stablehlo.negate %[[COPY_OUT]] : tensor<8xf32>
  // CHECK-NEXT: return %[[NEGATE_OUT]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %2 = stablehlo.negate %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: func private @xla.sdy.inlineable_callee(
// CHECK-SAME: %[[ARG:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ARG]], %[[ARG]] : tensor<8xf32>
// CHECK-NEXT: return %[[ADD]] : tensor<8xf32>
