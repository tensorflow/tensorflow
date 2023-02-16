// RUN: tf-opt -hlo-legalize-to-linalg -hlo-xla-runtime-sparsification %s | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = ["compressed"] }>

// CHECK-LABEL: func.func @mult_sparse_dense(
// CHECK-SAME:    %[[PTR:.*0]]: memref<?xindex>,
// CHECK-SAME:    %[[IDX:.*1]]: memref<?xindex>,
// CHECK-SAME:    %[[VAL:.*2]]: memref<?xf64>,
// CHECK-SAME:    %[[SPEC:.*3]]: !llvm.struct<(array<1 x i64>, array<3 x i64>)>
// CHECK-SAME:    %[[DENSE:.*4]]: memref<10xf64>) -> memref<10xf64> {
// CHECK-DAG:     %[[F0:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[A:.*]] = memref.alloc() {alignment = 64 : i64} : memref<10xf64>
// CHECK:         linalg.fill ins(%[[F0]] : f64) outs(%[[A]] : memref<10xf64>)
// CHECK:         %[[LO:.*]] = memref.load %[[PTR]][%[[C0]]] : memref<?xindex>
// CHECK:         %[[HI:.*]] = memref.load %[[PTR]][%[[C1]]] : memref<?xindex>
// CHECK:         scf.for %[[II:.*]] = %[[LO]] to %[[HI]] step %[[C1]] {
// CHECK:           %[[I:.*]] = memref.load %[[IDX]][%[[II]]] : memref<?xindex>
// CHECK:           %[[T0:.*]] = memref.load %[[VAL]][%[[II]]] : memref<?xf64>
// CHECK:           %[[T1:.*]] = memref.load %[[DENSE]][%[[I]]] : memref<10xf64>
// CHECK:           %[[T3:.*]] = arith.mulf %[[T0]], %[[T1]] : f64
// CHECK:           memref.store %[[T3]], %[[A]][%[[I]]] : memref<10xf64>
// CHECK:         }
// CHECK:         return %[[A]] : memref<10xf64>
// CHECK:       }
func.func @mult_sparse_dense(%arg0: tensor<10xf64, #SparseVector>,
                             %arg1: tensor<10xf64>)
			         -> tensor<10xf64> {
  %0 = mhlo.multiply %arg0, %arg1 : (tensor<10xf64, #SparseVector>,
                                     tensor<10xf64>) -> tensor<10xf64>
  return %0 : tensor<10xf64>
}
