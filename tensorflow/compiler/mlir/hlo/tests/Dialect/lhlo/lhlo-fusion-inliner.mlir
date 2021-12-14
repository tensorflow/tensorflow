// RUN: mlir-hlo-opt %s -lhlo-fusion-inliner -split-input-file | mlir-hlo-opt | FileCheck %s

// CHECK-LABEL: @fusion_after_codegen
// CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<3xi32>, %[[ARG3:.*]]: memref<?xf32>, %[[ARG4:.*]]: memref<?x?x?xf32>) -> memref<?x?x?xf32>
func @fusion_after_codegen(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<3xi32>, %arg3: memref<?xf32>, %arg4: memref<?x?x?xf32>) -> memref<?x?x?xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK:  %[[TMP0:.*]] = memref.dim %[[ARG4]], %[[C0]] : memref<?x?x?xf32>
  // CHECK:  %[[TMP1:.*]] = memref.dim %[[ARG4]], %[[C1]] : memref<?x?x?xf32>
  // CHECK:  %[[TMP2:.*]] = arith.muli %[[TMP0]], %[[TMP1]] : index
  // CHECK:  %[[TMP3:.*]] = memref.dim %[[ARG4]], %[[C2]] : memref<?x?x?xf32>
  // CHECK:  %[[TMP4:.*]] = arith.muli %[[TMP2]], %[[TMP3]] : index
  // CHECK:  scf.parallel (%[[ARG5:.*]]) = (%[[C0]]) to (%[[TMP4]]) step (%[[C1]]) {
  // CHECK:    %[[TMP5:.*]] = arith.muli %[[TMP3]], %[[TMP1]] : index
  // CHECK:    %[[TMP6:.*]] = arith.remui %[[ARG5]], %[[TMP5]] : index
  // CHECK:    %[[TMP7:.*]] = arith.remui %[[TMP6]], %[[TMP3]] : index
  // CHECK:    %[[TMP8:.*]] = memref.dim %[[ARG3]], %[[C0]] : memref<?xf32>
  // CHECK:    %[[TMP9:.*]] = arith.cmpi eq, %[[TMP8]], %[[C1]] : index
  // CHECK:    %[[TMP10:.*]] = select %[[TMP9]], %[[C0]], %[[TMP7]] : index
  // CHECK:    %[[TMP11:.*]] = memref.load %[[ARG0]][%[[TMP10]]] : memref<?xf32>
  // CHECK:    %[[TMP12:.*]] = memref.load %[[ARG1]][%[[TMP10]]] : memref<?xf32>
  // CHECK:    %[[TMP13:.*]] = arith.addf %[[TMP11]], %[[TMP12]] : f32
  // CHECK:    %[[TMP14:.*]] = memref.reinterpret_cast %[[ARG4]] to offset: [%[[C0]]], sizes: [%[[TMP4]]], strides: [%[[C1]]] : memref<?x?x?xf32> to memref<?xf32>
  // CHECK:    memref.store %[[TMP13]], %[[TMP14]][%[[ARG5]]] : memref<?xf32>
  // CHECK:    scf.yield
  // CHECK:  }
  "lmhlo.fusion"() ( {
    %0 = memref.dim %arg4, %c0 : memref<?x?x?xf32>
    %1 = memref.dim %arg4, %c1 : memref<?x?x?xf32>
    %2 = arith.muli %0, %1 : index
    %3 = memref.dim %arg4, %c2 : memref<?x?x?xf32>
    %4 = arith.muli %2, %3 : index
    scf.parallel (%arg5) = (%c0) to (%4) step (%c1) {
      %5 = arith.muli %3, %1 : index
      %6 = arith.remui %arg5, %5 : index
      %7 = arith.remui %6, %3 : index
      %8 = memref.dim %arg3, %c0 : memref<?xf32>
      %9 = arith.cmpi eq, %8, %c1 : index
      %10 = select %9, %c0, %7 : index
      %11 = memref.load %arg0[%10] : memref<?xf32>
      %12 = memref.load %arg1[%10] : memref<?xf32>
      %13 = arith.addf %11, %12 : f32
      %14 = memref.reinterpret_cast %arg4 to offset: [%c0], sizes: [%4], strides: [%c1] : memref<?x?x?xf32> to memref<?xf32>
      memref.store %13, %14[%arg5] : memref<?xf32>
      scf.yield
    }
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  return %arg4 : memref<?x?x?xf32>
}
