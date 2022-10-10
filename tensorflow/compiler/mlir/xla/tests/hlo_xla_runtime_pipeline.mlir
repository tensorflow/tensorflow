// RUN: xla-opt -split-input-file -hlo-xla-runtime-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @simple_add(
func.func @simple_add(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK: linalg.generic
  // CHECK: addf
  %0 = mhlo.add %arg0, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

// -----

#CSR = #sparse_tensor.encoding<{dimLevelType = [ "dense", "compressed" ]}>

// CHECK-LABEL: func.func @csr_abs_eltwise(
func.func @csr_abs_eltwise(%arg0: tensor<10x20xf32, #CSR>)
    -> tensor<10x20xf32, #CSR> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C10:.*]] = arith.constant 10 : index
  // CHECK-DAG:  %[[PTR:.*]] = call @sparsePointers0
  // CHECK-DAG:  %[[IDX:.*]] = call @sparseIndices0
  // CHECK-DAG:  %[[VAL:.*]] = call @sparseValuesF32
  // CHECK:      scf.for %[[I:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
  // CHECK:        %[[L:.*]] = memref.load %[[PTR]][%[[I]]] : memref<?xindex>
  // CHECK:        %[[A:.*]] = arith.addi %[[I]], %[[C1]] : index
  // CHECK:        %[[U:.*]] = memref.load %[[PTR]][%[[A]]] : memref<?xindex>
  // CHECK:        scf.for %[[JJ:.*]] = %[[L]] to %[[U]] step %[[C1]] {
  // CHECK:          %[[J:.*]] = memref.load %[[IDX]][%[[JJ]]] : memref<?xindex>
  // CHECK:          %[[V:.*]] = memref.load %[[VAL]][%[[JJ]]] : memref<?xf32>
  // CHECK:          math.absf %[[V]] : f32
  // CHECK:        }
  // CHECK:      }
  %0 = mhlo.abs %arg0 : tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}

// -----

#CSR = #sparse_tensor.encoding<{dimLevelType = [ "dense", "compressed" ]}>

// CHECK-LABEL: func.func @csr_gendot(
func.func @csr_gendot(%arg0: tensor<32x64xf64, #CSR>,
                      %arg1: tensor<64x32xf64>) -> tensor<32x32xf64> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG:  %[[PTR:.*]] = call @sparsePointers0
  // CHECK-DAG:  %[[IDX:.*]] = call @sparseIndices0
  // CHECK-DAG:  %[[VAL:.*]] = call @sparseValuesF64
  // CHECK:      scf.for %[[I:.*]] = %[[C0]] to %[[C32]] step %[[C1]] {
  // CHECK:        %[[L:.*]] = memref.load %[[PTR]][%[[I]]] : memref<?xindex>
  // CHECK:        %[[A:.*]] = arith.addi %[[I]], %[[C1]] : index
  // CHECK:        %[[U:.*]] = memref.load %[[PTR]][%[[A]]] : memref<?xindex>
  // CHECK:        scf.for %[[JJ:.*]] = %[[L]] to %[[U]] step %[[C1]] {
  // CHECK:          %[[J:.*]] = memref.load %[[IDX]][%[[JJ]]] : memref<?xindex>
  // CHECK:          %[[V:.*]] = memref.load %[[VAL]][%[[JJ]]] : memref<?xf64>
  // CHECK:          scf.for %[[K:.*]] = %[[C0]] to %[[C32]] step %[[C1]] {
  // CHECK:          }
  // CHECK:        }
  // CHECK:      }
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
                                      rhs_contracting_dimensions = [0]>,
    precision_config = [#mhlo<precision DEFAULT>,
                        #mhlo<precision DEFAULT>]}
    : (tensor<32x64xf64, #CSR>,
       tensor<64x32xf64>) -> tensor<32x32xf64>
  return %0 : tensor<32x32xf64>
}
