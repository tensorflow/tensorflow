// RUN: mlir-opt %s -linalg-lower-to-loops | FileCheck %s

// CHECK: #[[ID:.*]] = (d0) -> (d0)

func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %J = linalg.range %c0:%arg2:%c1 : !linalg.range
  %K = linalg.range %c0:%arg3:%c1 : !linalg.range
  %A = linalg.view %arg0[%I, %K] : !linalg.view<?x?xf32>
  %B = linalg.view %arg0[%K, %J] : !linalg.view<?x?xf32>
  %C = linalg.view %arg0[%I, %J] : !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return
}
// CHECK-LABEL: func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       CHECK: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       CHECK: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       CHECK: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       CHECK: affine.for %i0 = #[[ID]](%c0) to #[[ID]](%arg1) {
//       CHECK:   affine.for %i1 = #[[ID]](%c0) to #[[ID]](%arg2) {
//       CHECK:     affine.for %i2 = #[[ID]](%c0) to #[[ID]](%arg3) {
//   CHECK-DAG:       %[[a:.*]] = linalg.load %[[A]][%i0, %i2] : !linalg.view<?x?xf32>
//   CHECK-DAG:       %[[b:.*]] = linalg.load %[[B]][%i2, %i1] : !linalg.view<?x?xf32>
//       CHECK:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECK:       %[[c:.*]] = linalg.load %[[C]][%i0, %i1] : !linalg.view<?x?xf32>
//       CHECK:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:       linalg.store %[[res]], %[[C]][%i0, %i1] : !linalg.view<?x?xf32>

func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %J = linalg.range %c0:%arg2:%c1 : !linalg.range
  %2 = linalg.view %arg0[%I, %J] : !linalg.view<?x?xf32>
  %3 = linalg.view %arg0[%J] : !linalg.view<?xf32>
  %4 = linalg.view %arg0[%I] : !linalg.view<?xf32>
  linalg.matvec(%2, %3, %4) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// CHECK-LABEL: func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       CHECK: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       CHECK: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       CHECK: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       CHECK: affine.for %i0 = #[[ID]](%c0) to #[[ID]](%arg1) {
//       CHECK:   affine.for %i1 = #[[ID]](%c0) to #[[ID]](%arg2) {
//   CHECK-DAG:     %[[a:.*]] = linalg.load %[[A]][%i0, %i1] : !linalg.view<?x?xf32>
//   CHECK-DAG:     %[[b:.*]] = linalg.load %[[B]][%i1] : !linalg.view<?xf32>
//       CHECK:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECK:     %[[c:.*]] = linalg.load %[[C]][%i0] : !linalg.view<?xf32>
//       CHECK:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:     linalg.store %[[res]], %[[C]][%i0] : !linalg.view<?xf32>

func @dot(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %1 = linalg.view %arg0[%I] : !linalg.view<?xf32>
  %2 = linalg.view %arg0[%I] : !linalg.view<?xf32>
  %3 = linalg.view %arg0[] : !linalg.view<f32>
  linalg.dot(%1, %2, %3) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @dot(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       CHECK: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       CHECK: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       CHECK: %[[C:.*]] = linalg.view %arg0[] : !linalg.view<f32>
//       CHECK: affine.for %i0 = #[[ID]](%c0) to #[[ID]](%arg1) {
//   CHECK-DAG:   %[[a:.*]] = linalg.load %[[A]][%i0] : !linalg.view<?xf32>
//   CHECK-DAG:   %[[b:.*]] = linalg.load %[[B]][%i0] : !linalg.view<?xf32>
//       CHECK:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECK:   %[[c:.*]] = linalg.load %[[C]][] : !linalg.view<f32>
//       CHECK:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   linalg.store %[[res]], %[[C]][] : !linalg.view<f32>