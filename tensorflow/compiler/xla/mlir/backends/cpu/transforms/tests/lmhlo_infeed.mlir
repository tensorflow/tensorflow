// RUN: xla-cpu-opt %s -xla-lmhlo-to-cpu-runtime | FileCheck %s

// CHECK: func @cpu_infeed(
// CHECK:   %[[ARG0:[a-z0-9]+]]: memref<8xf32>
// CHECK: )
func.func @cpu_infeed(%arg0: memref<8xf32>) {
  // CHECK: call @[[INFEED:.*]](%[[ARG0]]) : (memref<8xf32>) -> ()
  "lmhlo.infeed"(%arg0) {config = "abc"} : (memref<8xf32>) -> ()
  return
}

// CHECK: func private @[[INFEED]](memref<8xf32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.cpu.infeed"}
