// RUN: tf-tfrt-opt %s -tf-cpurt-linalg-matmul-specialization | FileCheck %s

// CHECK-LABEL: @matmul_dynamic
func @matmul_dynamic(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
                     %arg2: memref<?x?xf32>) {
  // CHECK-DAG: %[[C0:.*]] = constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index

  // CHECK: %[[M:.*]] = memref.dim %arg0, %[[C0]]
  // CHECK: %[[K:.*]] = memref.dim %arg0, %[[C1]]
  // CHECK: %[[N:.*]] = memref.dim %arg1, %[[C1]]

  // CHECK: %[[M_ONE:.*]] = cmpi eq, %[[M]], %[[C1]]
  // CHECK: %[[N_ONE:.*]] = cmpi eq, %[[N]], %[[C1]]
  // CHECK: %[[M_NOT_ONE:.*]] = cmpi ne, %[[M]], %[[C1]]
  // CHECK: %[[N_NOT_ONE:.*]] = cmpi ne, %[[N]], %[[C1]]

  // CHECK: %[[IS_DOT:.*]] = and %[[M_ONE]], %[[N_ONE]]
  // CHECK: %[[IS_VECMAT:.*]] = and %[[M_ONE]], %[[N_NOT_ONE]]
  // CHECK: %[[IS_MATVEC:.*]] = and %[[N_ONE]], %[[M_NOT_ONE]]

  // CHECK: scf.if %[[IS_DOT]] {
  // CHECK: memref.reinterpret_cast %arg0 {{.*}} to memref<?xf32>
  // CHECK: memref.reinterpret_cast %arg1 {{.*}} to memref<?xf32>
  // CHECK: memref.reinterpret_cast %arg2 {{.*}} to memref<f32>
  // CHECK: linalg.dot
  // CHECK: } else

  // CHECK: scf.if %[[IS_VECMAT]] {
  // CHECK: memref.reinterpret_cast %arg0 {{.*}} to memref<?xf32>
  // CHECK: memref.reinterpret_cast %arg2 {{.*}} to memref<?xf32>
  // CHECK: linalg.vecmat
  // CHECK: } else

  // CHECK: scf.if %[[IS_MATVEC]] {
  // CHECK: memref.reinterpret_cast %arg1 {{.*}} to memref<?xf32>
  // CHECK: memref.reinterpret_cast %arg2 {{.*}} to memref<?xf32>
  // CHECK: linalg.matvec
  // CHECK: } else

  // CHECK: linalg.matmul {__tf_cpurt_specialized}
  // CHECK: }
  linalg.matmul ins(%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)
                outs(%arg2: memref<?x?xf32>)
  return
}

// CHECK-LABEL: @matmul_static_k
func @matmul_static_k(%arg0: memref<?x4xf32>, %arg1: memref<4x?xf32>,
                      %arg2: memref<?x?xf32>) {
  // CHECK-DAG: %[[C0:.*]] = constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index

  // CHECK: %[[M:.*]] = memref.dim %arg0, %[[C0]]
  // CHECK: %[[N:.*]] = memref.dim %arg1, %[[C1]]

  // CHECK: %[[M_ONE:.*]] = cmpi eq, %[[M]], %[[C1]]
  // CHECK: %[[N_ONE:.*]] = cmpi eq, %[[N]], %[[C1]]
  // CHECK: %[[M_NOT_ONE:.*]] = cmpi ne, %[[M]], %[[C1]]
  // CHECK: %[[N_NOT_ONE:.*]] = cmpi ne, %[[N]], %[[C1]]

  // CHECK: %[[IS_DOT:.*]] = and %[[M_ONE]], %[[N_ONE]]
  // CHECK: %[[IS_VECMAT:.*]] = and %[[M_ONE]], %[[N_NOT_ONE]]
  // CHECK: %[[IS_MATVEC:.*]] = and %[[N_ONE]], %[[M_NOT_ONE]]

  // CHECK: scf.if %[[IS_DOT]] {
  // CHECK: memref.reinterpret_cast %arg0 {{.*}} to memref<4xf32>
  // CHECK: memref.reinterpret_cast %arg1 {{.*}} to memref<4xf32>
  // CHECK: memref.reinterpret_cast %arg2 {{.*}} to memref<f32>
  // CHECK: linalg.dot
  // CHECK: } else

  // CHECK: scf.if %[[IS_VECMAT]] {
  // CHECK: memref.reinterpret_cast %arg0 {{.*}} to memref<4xf32>
  // CHECK: memref.reinterpret_cast %arg2 {{.*}} to memref<?xf32>
  // CHECK: linalg.vecmat
  // CHECK: } else

  // CHECK: scf.if %[[IS_MATVEC]] {
  // CHECK: memref.reinterpret_cast %arg1 {{.*}} to memref<4xf32>
  // CHECK: memref.reinterpret_cast %arg2 {{.*}} to memref<?xf32>
  // CHECK: linalg.matvec
  // CHECK: } else

  // CHECK: linalg.matmul {__tf_cpurt_specialized}
  // CHECK: }
  linalg.matmul ins(%arg0, %arg1: memref<?x4xf32>, memref<4x?xf32>)
                outs(%arg2: memref<?x?xf32>)
  return
}
