// RUN: sdy_opt %s -split-input-file -xla-sdy-stablehlo-export-pipeline="erase-manual-computations=true" 2>&1 | FileCheck %s

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @manual_comp_erase
// CHECK-SAME: %[[ARG0:.*]]: tensor<8xf32>
func.func @manual_comp_erase(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[COPY_IN:.*]] = mhlo.copy %[[ARG0]] {mhlo.sharding = "{replicated}"} : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @xla.sdy.inlineable_callee(%[[COPY_IN]]) {mhlo.sharding = "{replicated}"} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[COPY_OUT:.*]] = mhlo.copy %[[CALL]] {mhlo.sharding = "{replicated}"} : tensor<8xf32>
  // CHECK-NEXT: return %[[COPY_OUT]] : tensor<8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{}]>] manual_axes={} (%arg1: tensor<8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @xla.sdy.inlineable_callee(
// CHECK-SAME: %[[ARG:.*]]: tensor<8xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<8xf32> {mhlo.sharding = "{replicated}"}) {
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ARG]], %[[ARG]] : tensor<8xf32>
// CHECK-NEXT: return %[[ADD]] : tensor<8xf32>
