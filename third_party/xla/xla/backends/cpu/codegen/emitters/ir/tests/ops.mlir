// RUN: emitters_opt %s --split-input-file | FileCheck %s

func.func @load(%arg0: !xla_cpu.call_frame) -> tensor<32x32xf32> {
  %0 = xla_cpu.load %arg0, 0 : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @load(
// CHECK:   %[[ARG0:.+]]: !xla_cpu.call_frame
// CHECK: ) -> tensor<32x32xf32> {
// CHECK:   %[[LOAD:.+]] = xla_cpu.load %[[ARG0]], 0 : tensor<32x32xf32>
// CHECK:   return %[[LOAD]] : tensor<32x32xf32>
// CHECK: }

