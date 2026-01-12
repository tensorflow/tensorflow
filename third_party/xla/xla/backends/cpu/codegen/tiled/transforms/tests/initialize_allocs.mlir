// RUN: fusion_compiler_opt %s -xtile-cpu-initialize-allocs -cse | FileCheck %s

// CHECK-LABEL: @int_alloc
func.func @int_alloc() -> memref<5xi32> {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<5xi32>
  // CHECK-DAG: %[[C_HIGH:.*]] = arith.constant -1 : i32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM:.*]] = memref.dim %[[ALLOC]], %[[C0]] : memref<5xi32>
  // CHECK: scf.for %[[IDX:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
  // CHECK:   memref.store %[[C_HIGH]], %[[ALLOC]][%[[IDX]]] : memref<5xi32>
  // CHECK: }
  %0 = memref.alloc() : memref<5xi32>
  // CHECK: return %[[ALLOC]] : memref<5xi32>
  func.return %0 : memref<5xi32>
}

// CHECK-LABEL: @float_alloc
func.func @float_alloc() -> memref<5x10xf32> {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<5x10xf32>
  // CHECK-DAG: %[[NaN:.*]] = arith.constant 0x7FC00000 : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM_0:.*]] = memref.dim %[[ALLOC]], %[[C0]] : memref<5x10xf32>
  // CHECK: %[[DIM_1:.*]] = memref.dim %[[ALLOC]], %[[C1]] : memref<5x10xf32>
  // CHECK: scf.for %[[IDX_0:.*]] = %[[C0]] to %[[DIM_0]] step %[[C1]] {
  // CHECK:   scf.for %[[IDX_1:.*]] = %[[C0]] to %[[DIM_1]] step %[[C1]] {
  // CHECK:     memref.store %[[NaN]], %[[ALLOC]][%[[IDX_0]], %[[IDX_1]]] : memref<5x10xf32>
  // CHECK:   }
  // CHECK: }
  %0 = memref.alloc() : memref<5x10xf32>
  // CHECK: return %[[ALLOC]] : memref<5x10xf32>
  func.return %0 : memref<5x10xf32>
}
