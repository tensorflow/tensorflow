// RUN: xla_cpu_opt %s --xla-cpu-lower-trivial | FileCheck %s

func.func @call_frame_arg(%arg0: !xla_cpu.call_frame) {
  %0 = xla_cpu.load %arg0, 0 : tensor<32x32xf32>
  return
}

// CHECK-LABEL: @call_frame_arg(
// CHECK: %[[ARG0:.+]]: !llvm.ptr
// CHECK: ) {
// CHECK:   %[[ARGS_GEP:.+]] = llvm.getelementptr %[[ARG0]][3]
// CHECK:   %[[ARGS:.+]] = llvm.load %[[ARGS_GEP]]
// CHECK:   %[[ARG_GEP:.+]] = llvm.getelementptr %[[ARGS]][0]
// CHECK:   %[[ARG:.+]] = llvm.load %[[ARG_GEP]]
// CHECK:   return
// CHECK: }
