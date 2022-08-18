// RUN: xla-runtime-opt %s --xla-memref-aligned-allocations \
// RUN:   | FileCheck %s

// RUN: xla-runtime-opt %s --xla-memref-aligned-allocations=alignment=16 \
// RUN:   | FileCheck --check-prefix=ALIGN16 %s

// CHECK-LABEL: @aligned_alloc
// ALIGN16-LABEL: @aligned_alloc
func.func @aligned_alloc(%arg0: index) -> memref<?xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%arg0) {alignment = 64 : i64}
  // ALIGN16: %[[ALLOC:.*]] = memref.alloc(%arg0) {alignment = 32 : i64}
  %0 = memref.alloc(%arg0) { alignment = 32 : i64 } : memref<?xf32>
  // CHECK: return %[[ALLOC]]
  // ALIGN16: return %[[ALLOC]]
  return %0 : memref<?xf32>
}

// CHECK-LABEL: @unaligned_alloc
// ALIGN16-LABEL: @unaligned_alloc
func.func @unaligned_alloc(%arg0: index) -> memref<?xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%arg0) {alignment = 64 : i64}
  // ALIGN16: %[[ALLOC:.*]] = memref.alloc(%arg0) {alignment = 16 : i64}
  %0 = memref.alloc(%arg0) : memref<?xf32>
  // CHECK: return %[[ALLOC]]
  // ALIGN16: return %[[ALLOC]]
  return %0 : memref<?xf32>
}

