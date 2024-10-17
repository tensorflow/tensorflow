// RUN: xla_cpu_opt %s --xla-cpu-lower-trivial | FileCheck %s

func.func @call_frame_arg(%arg0: !xla_cpu.call_frame) {
  return
}

// CHECK-LABEL: @call_frame_arg(
// CHECK-SAME:   %[[ARG0:.+]]: !llvm.ptr
// CHECK-SAME: )
