// RUN: xla-opt -split-input-file -hlo-xla-runtime-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @simple_add(
func.func @simple_add(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK: linalg.generic
  // CHECK: addf
  %0 = mhlo.add %arg0, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

// -----

// TODO(ecg): bring back the Sparse tests once BufferResultsToOutParams is
// restricted to the main entry point.

//#CSR = #sparse_tensor.encoding<{dimLevelType = [ "dense", "compressed" ]}>

// NOCHECK-LABEL: func.func @csr_abs_eltwise(
//func.func @csr_abs_eltwise(%arg0: tensor<10x20xf32, #CSR>)
//    -> tensor<10x20xf32, #CSR> {
  // NOCHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // NOCHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // NOCHECK-DAG:  %[[C10:.*]] = arith.constant 10 : index
  // NOCHECK-DAG:  %[[PTR:.*]] = call @sparsePointers0
  // NOCHECK-DAG:  %[[IDX:.*]] = call @sparseIndices0
  // NOCHECK-DAG:  %[[VAL:.*]] = call @sparseValuesF32
  // NOCHECK:      scf.for %[[I:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
  // NOCHECK:        %[[L:.*]] = memref.load %[[PTR]][%[[I]]] : memref<?xindex>
  // NOCHECK:        %[[A:.*]] = arith.addi %[[I]], %[[C1]] : index
  // NOCHECK:        %[[U:.*]] = memref.load %[[PTR]][%[[A]]] : memref<?xindex>
  // NOCHECK:        scf.for %[[JJ:.*]] = %[[L]] to %[[U]] step %[[C1]] {
  // NOCHECK:          %[[J:.*]] = memref.load %[[IDX]][%[[JJ]]] : memref<?xindex>
  // NOCHECK:          %[[V:.*]] = memref.load %[[VAL]][%[[JJ]]] : memref<?xf32>
  // NOCHECK:          math.absf %[[V]] : f32
  // NOCHECK:        }
  // NOCHECK:      }
//  %0 = mhlo.abs %arg0 : tensor<10x20xf32, #CSR>
//  func.return %0 : tensor<10x20xf32, #CSR>
//}

// -----

//#CSR = #sparse_tensor.encoding<{dimLevelType = [ "dense", "compressed" ]}>

// NOCHECK-LABEL: func.func @csr_gendot(
//func.func @csr_gendot(%arg0: tensor<32x64xf64, #CSR>,
//                      %arg1: tensor<64x32xf64>) -> tensor<32x32xf64> {
  // NOCHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // NOCHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // NOCHECK-DAG:  %[[C32:.*]] = arith.constant 32 : index
  // NOCHECK-DAG:  %[[PTR:.*]] = call @sparsePointers0
  // NOCHECK-DAG:  %[[IDX:.*]] = call @sparseIndices0
  // NOCHECK-DAG:  %[[VAL:.*]] = call @sparseValuesF64
  // NOCHECK:      scf.for %[[I:.*]] = %[[C0]] to %[[C32]] step %[[C1]] {
  // NOCHECK:        %[[L:.*]] = memref.load %[[PTR]][%[[I]]] : memref<?xindex>
  // NOCHECK:        %[[A:.*]] = arith.addi %[[I]], %[[C1]] : index
  // NOCHECK:        %[[U:.*]] = memref.load %[[PTR]][%[[A]]] : memref<?xindex>
  // NOCHECK:        scf.for %[[JJ:.*]] = %[[L]] to %[[U]] step %[[C1]] {
  // NOCHECK:          %[[J:.*]] = memref.load %[[IDX]][%[[JJ]]] : memref<?xindex>
  // NOCHECK:          %[[V:.*]] = memref.load %[[VAL]][%[[JJ]]] : memref<?xf64>
  // NOCHECK:          scf.for %[[K:.*]] = %[[C0]] to %[[C32]] step %[[C1]] {
  // NOCHECK:          }
  // NOCHECK:        }
  // NOCHECK:      }
//  %0 = "mhlo.dot_general"(%arg0, %arg1) {
//    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
//                                      rhs_contracting_dimensions = [0]>,
//    precision_config = [#mhlo<precision DEFAULT>,
//                        #mhlo<precision DEFAULT>]}
//    : (tensor<32x64xf64, #CSR>,
//       tensor<64x32xf64>) -> tensor<32x32xf64>
//  return %0 : tensor<32x32xf64>
//}
