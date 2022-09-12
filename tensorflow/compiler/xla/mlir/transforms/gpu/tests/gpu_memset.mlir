// RUN: xla-gpu-opt %s --split-input-file -xla-gpu-to-gpu-runtime | FileCheck %s

// CHECK: func @gpu_memset_i32(
// CHECK:   %[[DST:[a-z0-9]+]]: memref<?xi32>
// CHECK: )
func.func @gpu_memset_i32(%dst: memref<?xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant 0 : i32
  %cst = arith.constant 0 : i32
  // CHECK: call @[[MEMSET:.*]](%[[DST]], %[[CST]])
  gpu.memset %dst, %cst : memref<?xi32>, i32
  return
}

// CHECK: func private @[[MEMSET]](memref<?xi32>, i32)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.memset"}

// -----

// CHECK: func @gpu_memset_f32(
// CHECK:   %[[DST:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_memset_f32(%dst: memref<?xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: call @[[MEMSET:.*]](%[[DST]], %[[CST]])
  gpu.memset %dst, %cst : memref<?xf32>, f32
  return
}

// CHECK: func private @[[MEMSET]](memref<?xf32>, f32)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.memset"}
