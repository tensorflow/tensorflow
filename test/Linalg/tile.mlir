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
//  TILE-2-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[sAi:.*]] = linalg.subview %arg0[%i0, %[[a]], %c1, %c0, %[[K]], %c1] : !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[N:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[sCi:.*]] = linalg.subview %arg2[%i0, %[[c]], %c1, %c0, %[[N]], %c1] : !linalg.view<?x?xf32>
//  TILE-2-NEXT:   linalg.matmul(%[[sAi]], %arg1, %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-02-LABEL: func @matmul(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?x?xf32>, %arg2: !linalg.view<?x?xf32>) {
//       TILE-02: %[[N:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?xf32>
//       TILE-02: linalg.for %i0 = %c0 to %[[N]] step %c2 {
//  TILE-02-NEXT:   %[[K:.*]] = linalg.dim %arg1, 0 : !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[sBj:.*]] = linalg.subview %arg1[%c0, %[[K]], %c1, %i0, %[[b]], %c1] : !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[M:.*]] = linalg.dim %arg2, 0 : !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[sCj:.*]] = linalg.subview %arg2[%c0, %[[M]], %c1, %i0, %[[c]], %c1] : !linalg.view<?x?xf32>
//  TILE-02-NEXT:   linalg.matmul(%arg0, %[[sBj]], %[[sCj]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-002-LABEL: func @matmul(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?x?xf32>, %arg2: !linalg.view<?x?xf32>) {
//       TILE-002: %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-002: linalg.for %i0 = %c0{{.*}} to %[[K]] step %c2 {
//  TILE-002-NEXT:   %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//  TILE-002-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-002-NEXT:   %[[sAj:.*]] = linalg.subview %arg0[%c0, %[[M]], %c1, %i0, %[[a]], %c1] : !linalg.view<?x?xf32>
//  TILE-002-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-002-NEXT:   %[[N:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?xf32>
//  TILE-002-NEXT:   %[[sBj:.*]] = linalg.subview %arg1[%i0, %[[b]], %c1, %c0, %[[N]], %c1] : !linalg.view<?x?xf32>
//  TILE-002-NEXT:   linalg.matmul(%[[sAj]], %[[sBj]], %arg2) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-234-LABEL: func @matmul(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?x?xf32>, %arg2: !linalg.view<?x?xf32>) {
//       TILE-234: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       TILE-234: %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-234: %[[N:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?xf32>
//       TILE-234:  linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//  TILE-234-NEXT:    linalg.for %i1 = %c0{{.*}} to %[[N]] step %c3 {
//  TILE-234-NEXT:      linalg.for %i2 = %c0{{.*}} to %[[K]] step %c4 {
//  TILE-234-NEXT:        %[[ai:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:        %[[ak:.*]] = affine.apply #[[UB2]](%i2)
//  TILE-234-NEXT:        %[[sAik:.*]] = linalg.subview %arg0[%i0, %[[ai]], %c1, %i2, %[[ak]], %c1] : !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[bk:.*]] = affine.apply #[[UB2]](%i2)
//  TILE-234-NEXT:        %[[bj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:        %[[sBkj:.*]] = linalg.subview %arg1[%i2, %[[bk]], %c1, %i1, %[[bj]], %c1] : !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[ci:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:        %[[cj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:        %[[sCij:.*]] = linalg.subview %arg2[%i0, %[[ci]], %c1, %i1, %[[cj]], %c1] : !linalg.view<?x?xf32>
//
//  TILE-234-NEXT:        linalg.matmul(%[[sAik]], %[[sBkj]], %[[sCij]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
  linalg.matvec(%arg0, %arg1, %arg2) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// TILE-2-LABEL: func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
//       TILE-2: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       TILE-2: linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//  TILE-2-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[N:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[sAi:.*]] = linalg.subview %arg0[%i0, %[[a]], %c1, %c0, %[[N]], %c1] : !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[sCi:.*]] = linalg.subview %arg2[%i0, %[[c]], %c1] : !linalg.view<?xf32>
//  TILE-2-NEXT:   linalg.matvec(%[[sAi]], %arg1, %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-02-LABEL: func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
//       TILE-02: %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-02: linalg.for %i0 = %c0{{.*}} to %[[K]] step %c2 {
//  TILE-02-NEXT:   %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[sAj:.*]] = linalg.subview %arg0[%c0, %[[M]], %c1, %i0, %[[a]], %c1] : !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[sBj:.*]] = linalg.subview %arg1[%i0, %[[b]], %c1] : !linalg.view<?xf32>
//       TILE-02:   linalg.matvec(%[[sAj]], %[[sBj]], %arg2) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-002-LABEL: func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
//   TILE-002-NOT: linalg.for

// TILE-234-LABEL: func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
//       TILE-234: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       TILE-234: %[[K:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       TILE-234:  linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//  TILE-234-NEXT:    linalg.for %i1 = %c0{{.*}} to %[[K]] step %c3 {
//  TILE-234-NEXT:      %[[ai:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:      %[[aj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:      %[[sAij:.*]] = linalg.subview %arg0[%i0, %[[ai]], %c1, %i1, %[[aj]], %c1] : !linalg.view<?x?xf32>
//  TILE-234-NEXT:      %[[bj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:      %[[sBj:.*]] = linalg.subview %arg1[%i1, %[[bj]], %c1] : !linalg.view<?xf32>
//  TILE-234-NEXT:      %[[ci:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:      %[[sCi:.*]] = linalg.subview %arg2[%i0, %[[ci]], %c1] : !linalg.view<?xf32>
//
//  TILE-234-NEXT:      linalg.matvec(%[[sAij]], %[[sBj]], %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// TILE-2-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//       TILE-2: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//       TILE-2: linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//  TILE-2-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[sAi:.*]] = linalg.subview %arg0[%i0, %[[a]], %c1] : !linalg.view<?xf32>
//  TILE-2-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[sBi:.*]] = linalg.subview %arg1[%i0, %[[b]], %c1] : !linalg.view<?xf32>
//  TILE-2-NEXT:   linalg.dot(%[[sAi]], %[[sBi]], {{.*}}) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>

// TILE-02-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//   TILE-02-NOT: linalg.for

// TILE-002-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//   TILE-002-NOT: linalg.for

// TILE-234-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//       TILE-234: %[[K:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//       TILE-234:  linalg.for %i0 = %c0{{.*}} to %[[K]] step %c2 {
//  TILE-234-NEXT:    %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:    %[[sAi:.*]] = linalg.subview %arg0[%i0, %[[a]], %c1] : !linalg.view<?xf32>
//  TILE-234-NEXT:    %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:    %[[sBi:.*]] = linalg.subview %arg1[%i0, %[[b]], %c1] : !linalg.view<?xf32>
//  TILE-234-NEXT:    linalg.dot(%[[sAi]], %[[sBi]], %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
