// RUN: tf-opt -split-input-file -hlo-xla-runtime-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @simple_add(
func.func @simple_add(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK: arith.addf
  %0 = mhlo.add %arg0, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

// -----

#CSR = #sparse_tensor.encoding<{dimLevelType = [ "dense", "compressed" ]}>

// CHECK-LABEL: func.func @csr_gendot(
// CHECK-SAME:    %[[PTR:.*0]]: memref<?xindex>,
// CHECK-SAME:    %[[IDX:.*1]]: memref<?xindex>,
// CHECK-SAME:    %[[VAL:.*2]]: memref<?xf64>,
// CHECK-SAME:    %[[SPEC:.*3]]: !llvm.struct<(array<2 x i64>, array<3 x i64>)>
// CHECK-SAME:    %[[DENSE:.*4]]: memref<64x32xf64>) -> memref<32x32xf64> {
func.func @csr_gendot(%arg0: tensor<32x64xf64, #CSR>,
                      %arg1: tensor<64x32xf64>) -> tensor<32x32xf64> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG:  %[[ALLOC:.*]] = memref.alloc()
  // CHECK:      scf.for %[[I:.*]] = %[[C0]] to %[[C32]] step %[[C1]] {
  // CHECK:        %[[L:.*]] = memref.load %[[PTR]][%[[I]]] : memref<?xindex>
  // CHECK:        %[[A:.*]] = arith.addi %[[I]], %[[C1]] : index
  // CHECK:        %[[U:.*]] = memref.load %[[PTR]][%[[A]]] : memref<?xindex>
  // CHECK:        scf.for %[[JJ:.*]] = %[[L]] to %[[U]] step %[[C1]] {
  // CHECK:          %[[J:.*]] = memref.load %[[IDX]][%[[JJ]]] : memref<?xindex>
  // CHECK:          %[[V:.*]] = memref.load %[[VAL]][%[[JJ]]] : memref<?xf64>
  // CHECK:          scf.for %[[K:.*]] = %[[C0]] to %[[C32]] step %[[C1]] {
  // CHECK:            %[[T1:.*]] = memref.load %[[ALLOC]][%[[I]], %[[K]]] : memref<32x32xf64>
  // CHECK:            %[[T2:.*]] = memref.load %[[DENSE]][%[[J]], %[[K]]] : memref<64x32xf64>
  // CHECK:            %[[T3:.*]] = arith.mulf %[[V]], %[[T2]] : f64
  // CHECK:            %[[T4:.*]] = arith.addf %[[T1]], %[[T3]] : f64
  // CHECK:            memref.store %[[T4]], %[[ALLOC]][%[[I]], %[[K]]] : memref<32x32xf64>
  // CHECK:          }
  // CHECK:        }
  // CHECK:      }
  // CHECK:      return %[[ALLOC]] : memref<32x32xf64>
  // CHECK:    }
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1],
                                      rhs_contracting_dimensions = [0]>,
    precision_config = [#mhlo<precision DEFAULT>,
                        #mhlo<precision DEFAULT>]}
    : (tensor<32x64xf64, #CSR>,
       tensor<64x32xf64>) -> tensor<32x32xf64>
  return %0 : tensor<32x32xf64>
}
