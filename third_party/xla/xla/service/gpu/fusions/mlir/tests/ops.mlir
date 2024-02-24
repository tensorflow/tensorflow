// RUN: mlir_fusions_opt %s -canonicalize | FileCheck %s

module {
  func.func @shared_and_sync() -> (tensor<2xf32>, tensor<2xf32>) {
    %shared1 = xla_gpu.allocate_shared : tensor<2xf32>
    %shared2 = xla_gpu.allocate_shared : tensor<2xf32>
    %sync:2 = xla_gpu.sync_threads %shared1, %shared2
      : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    return %sync#0, %sync#1 : tensor<2xf32>, tensor<2xf32>
  }
}

// CHECK: @shared_and_sync
// CHECK-NEXT: allocate_shared
// CHECK-NEXT: allocate_shared
// CHECK-NEXT: sync_threads
// CHECK-NEXT: return
