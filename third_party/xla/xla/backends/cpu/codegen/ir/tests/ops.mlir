// RUN: xla_cpu_opt %s | xla_cpu_opt | FileCheck %s

func.func @call_frame_arg(%arg0: !xla_cpu.call_frame) -> tensor<32x32xf32> {
  %0 = xla_cpu.load %arg0, 0 : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @call_frame_arg(
// CHECK:   %[[ARG0:.+]]: !xla_cpu.call_frame
// CHECK: ) -> tensor<32x32xf32> {
// CHECK:   %[[LOAD:.+]] = xla_cpu.load %[[ARG0]], 0 : tensor<32x32xf32>
// CHECK:   return %[[LOAD]] : tensor<32x32xf32>
// CHECK: }
