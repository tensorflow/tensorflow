// RUN: xla_cpu_opt %s | FileCheck %s

func.func @call_frame_arg(%arg0: !xla_cpu.call_frame) {
  return
}

// CHECK-LABEL: @call_frame_arg(
// CHECK-SAME:   %[[ARG0:.+]]: !xla_cpu.call_frame
// CHECK-SAME: )
