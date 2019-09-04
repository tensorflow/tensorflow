// RUN: mlir-opt %s -linalg-fusion -linalg-fusion-tile-sizes=16 -cse | mlir-opt -linalg-fusion -linalg-fusion-tile-sizes=8 | FileCheck %s

func @f1(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%C, %D, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return %E : !linalg.view<?x?xf32>
}
// CHECK-LABEL: func @f1
//   CHECK-DAG: %[[c8:.*]] = constant 8
//   CHECK-DAG: %[[c16:.*]] = constant 16
//       CHECK:   loop.for %{{.*}} step %[[c16]] {
//       CHECK:     loop.for %{{.*}} %[[c8]] {
//       CHECK:       linalg.matmul
//       CHECK:       linalg.matmul