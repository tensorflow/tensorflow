// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2 | FileCheck %s -check-prefix=TILE-2
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,2 | FileCheck %s -check-prefix=TILE-02
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,0,2 | FileCheck %s -check-prefix=TILE-002
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 | FileCheck %s -check-prefix=TILE-234

//   TILE-2-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
//  TILE-02-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-002-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-234-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-234-DAG: #[[UB1:.*]] = (d0) -> (d0 + 3)
// TILE-234-DAG: #[[UB2:.*]] = (d0) -> (d0 + 4)

func @matmul(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?x?xf32>, %arg2: !linalg.view<?x?xf32>) {
  linalg.matmul(%arg0, %arg1, %arg2) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return
}
// TILE-2-LABEL: func @matmul(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?x?xf32>, %arg2: !linalg.view<?x?xf32>) {
//       TILE-2: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       TILE-2: linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//       TILE-2:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-2:   %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-2:   %[[sAi:.*]] = linalg.subview %arg0[%i0, %[[a]], %c1, %c0, %[[K]], %c1] : !linalg.view<?x?xf32>
//       TILE-2:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-2:   %[[N:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?xf32>
//       TILE-2:   %[[sCi:.*]] = linalg.subview %arg2[%i0, %[[c]], %c1, %c0, %[[N]], %c1] : !linalg.view<?x?xf32>
//       TILE-2:   linalg.matmul(%[[sAi]], %arg1, %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-02-LABEL: func @matmul(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?x?xf32>, %arg2: !linalg.view<?x?xf32>) {
//       TILE-02: %[[N:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?xf32>
//       TILE-02: linalg.for %i0 = %c0 to %[[N]] step %c2 {
//       TILE-02:   %[[K:.*]] = linalg.dim %arg1, 0 : !linalg.view<?x?xf32>
//       TILE-02:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-02:   %[[sBj:.*]] = linalg.subview %arg1[%c0, %[[K]], %c1, %i0, %[[b]], %c1] : !linalg.view<?x?xf32>
//       TILE-02:   %[[M:.*]] = linalg.dim %arg2, 0 : !linalg.view<?x?xf32>
//       TILE-02:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-02:   %[[sCj:.*]] = linalg.subview %arg2[%c0, %[[M]], %c1, %i0, %[[c]], %c1] : !linalg.view<?x?xf32>
//       TILE-02:   linalg.matmul(%arg0, %[[sBj]], %[[sCj]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-002-LABEL: func @matmul(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?x?xf32>, %arg2: !linalg.view<?x?xf32>) {
//       TILE-002: %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-002: linalg.for %i0 = %c0{{.*}} to %[[K]] step %c2 {
//       TILE-002:   %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       TILE-002:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-002:   %[[sAj:.*]] = linalg.subview %arg0[%c0, %[[M]], %c1, %i0, %[[a]], %c1] : !linalg.view<?x?xf32>
//       TILE-002:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-002:   %[[N:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?xf32>
//       TILE-002:   %[[sBj:.*]] = linalg.subview %arg1[%i0, %[[b]], %c1, %c0, %[[N]], %c1] : !linalg.view<?x?xf32>
//       TILE-002:   linalg.matmul(%[[sAj]], %[[sBj]], %arg2) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-234-LABEL: func @matmul(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?x?xf32>, %arg2: !linalg.view<?x?xf32>) {
//       TILE-234: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       TILE-234: %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-234: %[[N:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?xf32>
//       TILE-234:  linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//       TILE-234:    linalg.for %i1 = %c0{{.*}} to %[[N]] step %c3 {
//       TILE-234:      linalg.for %i2 = %c0{{.*}} to %[[K]] step %c4 {
//       TILE-234:        %[[ai:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-234:        %[[ak:.*]] = affine.apply #[[UB2]](%i2)
//       TILE-234:        %[[sAik:.*]] = linalg.subview %arg0[%i0, %[[ai]], %c1, %i2, %[[ak]], %c1] : !linalg.view<?x?xf32>
//       TILE-234:        %[[bk:.*]] = affine.apply #[[UB2]](%i2)
//       TILE-234:        %[[bj:.*]] = affine.apply #[[UB1]](%i1)
//       TILE-234:        %[[sBkj:.*]] = linalg.subview %arg1[%i2, %[[bk]], %c1, %i1, %[[bj]], %c1] : !linalg.view<?x?xf32>
//       TILE-234:        %[[ci:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-234:        %[[cj:.*]] = affine.apply #[[UB1]](%i1)
//       TILE-234:        %[[sCij:.*]] = linalg.subview %arg2[%i0, %[[ci]], %c1, %i1, %[[cj]], %c1] : !linalg.view<?x?xf32>
//
//       TILE-234:        linalg.matmul(%[[sAik]], %[[sBkj]], %[[sCij]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
  linalg.matvec(%arg0, %arg1, %arg2) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// TILE-2-LABEL: func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
//       TILE-2: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       TILE-2: linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//       TILE-2:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-2:   %[[N:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-2:   %[[sAi:.*]] = linalg.subview %arg0[%i0, %[[a]], %c1, %c0, %[[N]], %c1] : !linalg.view<?x?xf32>
//       TILE-2:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-2:   %[[sCi:.*]] = linalg.subview %arg2[%i0, %[[c]], %c1] : !linalg.view<?xf32>
//       TILE-2:   linalg.matvec(%[[sAi]], %arg1, %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-02-LABEL: func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
//       TILE-02: %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-02: linalg.for %i0 = %c0{{.*}} to %[[K]] step %c2 {
//       TILE-02:   %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       TILE-02:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-02:   %[[sAj:.*]] = linalg.subview %arg0[%c0, %[[M]], %c1, %i0, %[[a]], %c1] : !linalg.view<?x?xf32>
//       TILE-02:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-02:   %[[sBj:.*]] = linalg.subview %arg1[%i0, %[[b]], %c1] : !linalg.view<?xf32>
//       TILE-02:   linalg.matvec(%[[sAj]], %[[sBj]], %arg2) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-002-LABEL: func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
//   TILE-002-NOT: linalg.for

// TILE-234-LABEL: func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
//       TILE-234: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       TILE-234: %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-234:  linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//       TILE-234:    linalg.for %i1 = %c0{{.*}} to %[[K]] step %c3 {
//       TILE-234:      %[[ai:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-234:      %[[aj:.*]] = affine.apply #[[UB1]](%i1)
//       TILE-234:      %[[sAij:.*]] = linalg.subview %arg0[%i0, %[[ai]], %c1, %i1, %[[aj]], %c1] : !linalg.view<?x?xf32>
//       TILE-234:      %[[bj:.*]] = affine.apply #[[UB1]](%i1)
//       TILE-234:      %[[sBj:.*]] = linalg.subview %arg1[%i1, %[[bj]], %c1] : !linalg.view<?xf32>
//       TILE-234:      %[[ci:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-234:      %[[sCi:.*]] = linalg.subview %arg2[%i0, %[[ci]], %c1] : !linalg.view<?xf32>
//
//       TILE-234:      linalg.matvec(%[[sAij]], %[[sBj]], %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// TILE-2-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//       TILE-2: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//       TILE-2: linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//       TILE-2:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-2:   %[[sAi:.*]] = linalg.subview %arg0[%i0, %[[a]], %c1] : !linalg.view<?xf32>
//       TILE-2:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-2:   %[[sBi:.*]] = linalg.subview %arg1[%i0, %[[b]], %c1] : !linalg.view<?xf32>
//       TILE-2:   linalg.dot(%[[sAi]], %[[sBi]], {{.*}}) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>

// TILE-02-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//   TILE-02-NOT: linalg.for

// TILE-002-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//   TILE-002-NOT: linalg.for

// TILE-234-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//       TILE-234: %[[K:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//       TILE-234:  linalg.for %i0 = %c0{{.*}} to %[[K]] step %c2 {
//       TILE-234:    %[[a:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-234:    %[[sAi:.*]] = linalg.subview %arg0[%i0, %[[a]], %c1] : !linalg.view<?xf32>
//       TILE-234:    %[[b:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-234:    %[[sBi:.*]] = linalg.subview %arg1[%i0, %[[b]], %c1] : !linalg.view<?xf32>
//       TILE-234:    linalg.dot(%[[sAi]], %[[sBi]], %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
