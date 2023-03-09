// RUN: mlir-hlo-opt %s -hlo-legalize-shape-computations -split-input-file | FileCheck %s

 // CHECK-LABEL: func @get_dimension_size
func.func @get_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i32>) {
  %1 = "mhlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[FROM:.+]] = tensor.from_elements %[[IDX]]
// CHECK: return %[[FROM]] : tensor<i32>

// -----

 // CHECK-LABEL: func @reshape_dimension_size
func.func @reshape_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<1xi32>) {
  %0 = "mhlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %1 = "mhlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[FROM:.+]] = tensor.from_elements %[[IDX]]
// CHECK: return %[[FROM]] : tensor<1xi32>

// -----

// CHECK-LABEL: func @multiply_dimension_size
func.func @multiply_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i32>) {
  %0 = mhlo.constant dense<2> : tensor<i32>
  %1 = "mhlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %2 = "mhlo.multiply"(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}


// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[C2:.+]] = arith.constant 2
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[MUL:.+]] = arith.muli %[[IDX]], %[[C2]]
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[MUL]]
// CHECK: return %[[RES]]

// -----

// CHECK-LABEL: func @concat_dimension_size
func.func @concat_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<2xi32>) {
  %0 = "mhlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %1 = "mhlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
  %2 = mhlo.constant dense<2> : tensor<1xi32>
  %3 = "mhlo.concatenate"(%1, %2) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %3 : tensor<2xi32>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[C2:.+]] = arith.constant 2
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[IDX]], %[[C2]]
// CHECK: return %[[RES]]

