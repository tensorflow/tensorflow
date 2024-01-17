// RUN: xla-gpu-opt %s --split-input-file -xla-gpu-to-gpu-runtime | FileCheck %s

// CHECK: func @gpu_memcpy_d2d(
// CHECK:   %[[DST:[a-z0-9]+]]: memref<?xf32>,
// CHECK:   %[[SRC:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_memcpy_d2d(%dst: memref<?xf32>, %src: memref<?xf32>) {
  // CHECK: call @[[MEMCPY:.*]](%[[DST]], %[[SRC]])
  gpu.memcpy %dst, %src : memref<?xf32>, memref<?xf32>
  return
}

// CHECK: func private @[[MEMCPY]](memref<?xf32>, memref<?xf32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.memcpy.d2d"}

// -----

// CHECK: func @gpu_memcpy_h2d(
// CHECK:   %[[DST:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_memcpy_h2d(%dst: memref<?xf32>, %dim: index) {
  // CHECK: %[[SRC:.*]] = memref.alloca
  %src = memref.alloca(%dim) : memref<?xf32>
  // CHECK: call @[[MEMCPY:.*]](%[[DST]], %[[SRC]])
  gpu.memcpy %dst, %src : memref<?xf32>, memref<?xf32>
  return
}

// CHECK: func private @[[MEMCPY]](memref<?xf32>, memref<?xf32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.memcpy.h2d"}

// -----

// CHECK: func @gpu_memcpy_d2h(
// CHECK:   %[[SRC:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_memcpy_d2h(%src: memref<?xf32>, %dim: index) {
  // CHECK: %[[DST:.*]] = memref.alloca
  %dst = memref.alloca(%dim) : memref<?xf32>
  // CHECK: call @[[MEMCPY:.*]](%[[DST]], %[[SRC]])
  gpu.memcpy %dst, %src : memref<?xf32>, memref<?xf32>
  return
}

// CHECK: func private @[[MEMCPY]](memref<?xf32>, memref<?xf32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.memcpy.d2h"}
