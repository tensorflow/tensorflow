// RUN: xla-gpu-opt %s -xla-lmhlo-to-gpu-runtime | FileCheck %s

// CHECK: func @gpu_infeed(
// CHECK:   %[[ARG0:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_infeed(%arg0: memref<?xf32>) {
  // CHECK: call @[[INFEED:.*]](%[[ARG0]])
  // CHECK-SAME: {config = "abc"} : (memref<?xf32>) -> ()
  "lmhlo.infeed"(%arg0) {config = "abc"} : (memref<?xf32>) -> ()
  return
}

// CHECK: func private @[[INFEED]](memref<?xf32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.infeed"}
