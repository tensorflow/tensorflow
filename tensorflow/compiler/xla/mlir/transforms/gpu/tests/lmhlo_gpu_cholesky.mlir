// RUN: xla-gpu-opt %s -xla-lmhlo-gpu-to-gpu-runtime | FileCheck %s

// CHECK: @compute(
// CHECK:   %[[ARG0:[a-z0-9]+]]: memref<4x4xi32>
// CHECK:   %[[ARG1:[a-z0-9]+]]: memref<4x4xi32>
// CHECK:   %[[ARG2:[a-z0-9]+]]: memref<4x4xi32>
// CHECK:   %[[ARG3:[a-z0-9]+]]: memref<4x4xi32>
// CHECK: )
func.func @compute(%operand: memref<4x4xi32>, %a: memref<4x4xi32>,
                   %workspace: memref<4x4xi32>, %info: memref<4x4xi32>) {

  // CHECK: call @[[CHOLESKY:.*]](%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]])
  // CHECK-SAME: batch_size = 1 : i64
  // CHECK-SAME: is_lower = true
  // CHECK-SAME: n = 4 : i64
  "lmhlo_gpu.cholesky"(%operand, %a, %workspace, %info) {
    batch_size = 1 : i64,
    is_lower = true,
    n = 4 : i64
  } : (memref<4x4xi32>, memref<4x4xi32>, memref<4x4xi32>, memref<4x4xi32>) -> ()

  // CHECK-NEXT: return
  func.return
}

// CHECK: func private @[[CHOLESKY]](memref<4x4xi32>, memref<4x4xi32>,
// CHECK-SAME:                       memref<4x4xi32>, memref<4x4xi32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.cholesky"}
