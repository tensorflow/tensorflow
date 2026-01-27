// RUN: fusion_compiler_opt %s -xtile-cpu-initialize-allocs -canonicalize \
// RUN: | FileCheck %s

// CHECK-LABEL: @int_alloc
func.func @int_alloc() -> memref<5xi32> {
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<5xi32>
  // CHECK-DAG: %[[C_HIGH:.*]] = arith.constant -1 : i32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
  // CHECK-DAG: %[[C20:.*]] = arith.constant 20 : index
  // CHECK: %[[PTR_IDX:.*]] = memref.extract_aligned_pointer_as_index %[[ALLOC]]
  // CHECK: %[[PTR_INT:.*]] = arith.index_cast %[[PTR_IDX]]
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[PTR_INT]]
  // CHECK: call @__msan_unpoison(%[[PTR]], %[[C20]]) : (!llvm.ptr, index) -> ()
  // CHECK: scf.for %[[IDX:.*]] = %[[C0]] to %[[C5]] step %[[C1]] {
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
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
  // CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
  // CHECK-DAG: %[[C200:.*]] = arith.constant 200 : index
  // CHECK: %[[PTR_IDX:.*]] = memref.extract_aligned_pointer_as_index %[[ALLOC]]
  // CHECK: %[[PTR_INT:.*]] = arith.index_cast %[[PTR_IDX]]
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[PTR_INT]]
  // CHECK: call @__msan_unpoison(%[[PTR]], %[[C200]]) : (!llvm.ptr, index) -> ()
  // CHECK: scf.for %[[IDX_0:.*]] = %[[C0]] to %[[C5]] step %[[C1]] {
  // CHECK:   scf.for %[[IDX_1:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
  // CHECK:     memref.store %[[NaN]], %[[ALLOC]][%[[IDX_0]], %[[IDX_1]]] : memref<5x10xf32>
  // CHECK:   }
  // CHECK: }
  %0 = memref.alloc() : memref<5x10xf32>
  // CHECK: return %[[ALLOC]] : memref<5x10xf32>
  func.return %0 : memref<5x10xf32>
}
