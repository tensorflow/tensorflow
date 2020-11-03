// RUN: xla-mlir-gpu-opt --mlir-gpu-store-forwarding %s | FileCheck %s

// CHECK-LABEL: @forward
func @forward() -> f32 {
  %0 = alloc() : memref<1024xf32>
  %c42 = constant 24 : index
  // CHECK: %[[CST:.*]] = constant 1.0
  %c1 = constant 1.0 : f32
  store %c1, %0[%c42] : memref<1024xf32>
  // CHECK-NOT: load
  %1 = load %0[%c42] : memref<1024xf32>
  // CHECK: return %[[CST]]
  return %1 : f32
}

// CHECK-LABEL: @forward_alloca
func @forward_alloca() -> f32 {
  %0 = alloca() : memref<1024xf32>
  %c42 = constant 24 : index
  // CHECK: %[[CST:.*]] = constant 1.0
  %c1 = constant 1.0 : f32
  store %c1, %0[%c42] : memref<1024xf32>
  // CHECK-NOT: load
  %1 = load %0[%c42] : memref<1024xf32>
  // CHECK: return %[[CST]]
  return %1 : f32
}

// CHECK-LABEL: @wrong_index
func @wrong_index() -> f32 {
  %0 = alloc() : memref<1024xf32>
  %c42 = constant 24 : index
  %c12 = constant 12 : index
  %c1 = constant 1.0 : f32
  store %c1, %0[%c42] : memref<1024xf32>
  // CHECK: %[[RES:.*]] = load
  %1 = load %0[%c12] : memref<1024xf32>
  // CHECK: return %[[RES]]
  return %1 : f32
}

// CHECK-LABEL: @wrong_memref
func @wrong_memref() -> f32 {
  %0 = alloc() : memref<1024xf32>
  %1 = alloc() : memref<1024xf32>
  %c42 = constant 24 : index
  %c1 = constant 1.0 : f32
  store %c1, %0[%c42] : memref<1024xf32>
  // CHECK: %[[RES:.*]] = load
  %2 = load %1[%c42] : memref<1024xf32>
  // CHECK: return %[[RES]]
  return %2 : f32
}

// CHECK-LABEL: @with_parallel_loop
func @with_parallel_loop() {
  %0 = alloc() : memref<1024xf32>
  %c0 = constant 0 : index
  %c42 = constant 24 : index
  %c1 = constant 1 : index
  // CHECK: %[[CST:.*]] = constant 1.100000e+01 : f32
  %c11 = constant 1.100000e+01 : f32
  store %c11, %0[%c42] : memref<1024xf32>
  // CHECK: scf.parallel
  scf.parallel (%i0) = (%c0) to (%c42) step (%c1) {
    // CHECK-NOT: load
    %1 = load %0[%c42] : memref<1024xf32>
    // CHECK-NEXT: store %[[CST]]
    store %1, %0[%c0] : memref<1024xf32>
  }
  return
}
