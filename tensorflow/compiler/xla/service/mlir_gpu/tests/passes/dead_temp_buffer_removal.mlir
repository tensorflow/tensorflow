// RUN: xla-mlir-gpu-opt --mlir-gpu-dead-temp-buffer-removal %s | FileCheck %s

// CHECK-LABEL: @dead
func @dead() {
  // CHECK-NOT: alloc
  %0 = alloc() : memref<42xi32>
  %c0 = constant 0 : i32
  %c12 = constant 12 : index
  // CHECK-NOT: store
  store %c0, %0[%c12] : memref<42xi32>
  return
}

// CHECK-LABEL: @dead_alloca
func @dead_alloca() {
  // CHECK-NOT: alloca
  %0 = alloc() : memref<42xi32>
  %c0 = constant 0 : i32
  %c12 = constant 12 : index
  // CHECK-NOT: store
  store %c0, %0[%c12] : memref<42xi32>
  return
}

// CHECK-LABEL: @dead_load
func @dead_load() {
  // CHECK-NOT: alloc
  %0 = alloc() : memref<42xi32>
  %c0 = constant 0 : i32
  %c12 = constant 12 : index
  store %c0, %0[%c12] : memref<42xi32>
  %1 = load %0[%c12] : memref<42xi32>
  return
}

// CHECK-LABEL: @used_load
func @used_load() -> i32 {
  // CHECK: alloc
  %0 = alloc() : memref<42xi32>
  %c0 = constant 0 : i32
  %c12 = constant 12 : index
  store %c0, %0[%c12] : memref<42xi32>
  %1 = load %0[%c12] : memref<42xi32>
  return %1 : i32
}

// CHECK-LABEL: @dead_subview
func @dead_subview() {
  // CHECK-NOT: alloc
  %0 = alloc() : memref<42xi32>
  %c0 = constant 0 : i32
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %c12 = constant 12 : index
  store %c0, %0[%c12] : memref<42xi32>
  %1 = subview %0[%c12][%c4][%c1] : memref<42xi32> to memref<?xi32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>
  return
}

// CHECK-LABEL: @used_subview
func @used_subview() -> i32 {
  // CHECK: alloc
  %0 = alloc() : memref<42xi32>
  %c0 = constant 0 : i32
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %c12 = constant 12 : index
  store %c0, %0[%c12] : memref<42xi32>
  %1 = subview %0[%c12][%c4][%c1] : memref<42xi32> to memref<?xi32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>
  %2 = load %1[%c1] : memref<?xi32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>
  return %2 : i32
}
